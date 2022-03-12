"""
Software for running experiments and collecting data from Egg Counter
devices developed in the Phillips Lab at the University of Oregon

Author: Cody Jarrett
Organization: Phillips Lab, Institute of Ecology and Evolution,
              University of Oregon
"""
import os
import sys
import csv
import time
import serial
import struct
import datetime
import cv2 as cv
import numpy as np
import configparser
import egg_vid_get_main_gui
import egg_vid_get_meta_gui
import egg_vid_get_functions as funcs
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow


VER_STR = "v2.1"

global dac_setting
dac_setting = 0

RIG_NAME = "Coruscant"
MAIN_DIR = "/media/birdname/Elements/ec_main_dir/"
DEFAULT_VOLTAGE = 1550

EXPERIMENTS_DIR = MAIN_DIR + "experiment_data/"

SERIAL_PATH = "/dev/ttyUSB0"
BAUD = 115200
TIMEOUT = 1
THERMISTOR_CONSTANTS = [1.1279e-3, 2.3429e-4, 8.7298e-8]
ADC_SAMPLE_NUM = 25
ADC_SAMPLE_PERIOD = 5  # Seconds

MIN_TEMP = 0
MAX_TEMP = 50
DEFAULT_TEMPERATURE = "20C"

MAX_CONTOURS = 6
BINARY_THRESH = 15
CNT_AREA_THRESH = 20

IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_GET_NUM = 10
TIMER_PRECISION = 5

VIDEO_HOURS = 12
SEC_PER_VIDEO = int(VIDEO_HOURS * 60 * 60)

CNT_EXT = "_contours.csv"

try:
    with open(
        MAIN_DIR + "temp_profiles/" + DEFAULT_TEMPERATURE + ".txt"
    ) as file:
        dac_setting = file.read().strip()
    print(f"\nDefault temperature: {DEFAULT_TEMPERATURE}")
    print(f"Voltage: {dac_setting}")
except Exception:
    print(
        f"\nCould not read {DEFAULT_TEMPERATURE} file and set "
        "initial DAC voltage. Does the file exist?"
    )

config = configparser.ConfigParser()
config.read(MAIN_DIR + "egg_counter_config.ini")

runtime_options = []
for key in config["Runtimes"]:
    val = config["Runtimes"][f"{key}"]
    runtime_options.append(val)

exposure_options = []
for key in config["CamExposures"]:
    val = config["CamExposures"][f"{key}"]
    exposure_options.append(val)

temp_profiles = []
for profile in os.listdir(MAIN_DIR + "temp_profiles/"):
    name, ext = os.path.splitext(profile)
    temp_profiles.append(name)

meta_fields = [
    "FileName",
    "StartDate",
    "StartTime",
    "RunTime",
    "EndDate",
    "TempProfile",
    "ChipNum",
    "RigName",
    "Purpose",
    "SetABacteria",
    "SetAWorms",
    "SetBBacteria",
    "SetBWorms",
]

cnt_header = [
    "VidNum",
    "VidTime",
    "FrameNum",
    "X",
    "Y",
    "Area",
    "Perimeter",
    "Aspect",
    "Extent",
    "Pe/Ex",
    "Ar/Ex",
    "As/Ex",
]

set_a_worm_position_names = [f"SetAWorm{i}" for i in range(1, 17)]
set_b_worm_position_names = [f"SetBWorm{i}" for i in range(17, 33)]
meta_fields.extend(set_a_worm_position_names)
meta_fields.extend(set_b_worm_position_names)

global experiment_flag
experiment_flag = False

global experiment_name
experiment_name = ""

temp_collect_flag = False
meta_vals_temp_list = []

global brightness, contrast, gamma, sharpness, auto_exposure, exposure
brightness = -54
contrast = 87
gamma = 100
sharpness = 1
auto_exposure = 3
exposure = 625


# The ArduinoThread class is a QThread that
# communicates with an Arduino over a serial connection. It has a voltage attribute, a
# temp_data_file_str attribute, and a ser attribute. The voltage attribute is an int that
# represents the voltage that the Arduino should output. The temp_data_file_str attribute
# is a string that represents the file path to the temperature data file. The ser attribute
# is a serial object that represents the serial connection to the Arduino.
class ArduinoThread(QThread):
    def __init__(self, voltage, temp_data_file_str, parent=None):
        super(ArduinoThread, self).__init__(parent)

        self.voltage = int(voltage)
        self.temp_data_file_str = temp_data_file_str
        self.ser = serial.Serial(SERIAL_PATH, BAUD, timeout=TIMEOUT)
        time.sleep(2)

    def close_connection(self):
        self.ser.close()

    def read_temp(self):
        readings_list = list(range(ADC_SAMPLE_NUM))
        for i in range(ADC_SAMPLE_NUM):
            self.ser.write("R".encode("ascii"))
            line = self.ser.readline().decode("ascii").strip()
            readings_list[i] = int(line)
            time.sleep(0.001)
        avg_reading = sum(readings_list) / ADC_SAMPLE_NUM
        # Divide by 1000 to convert from mV to V
        volts = avg_reading / 1000
        res = volts * 10000
        return self.temp_convert(res), volts

    def set_dac_voltage(self, value):
        self.ser.write("S".encode("ascii"))
        packedValue = struct.pack("<l", value)
        self.ser.write(packedValue)
        print(f"Voltage: {value}")

    def temp_convert(self, res):
        term2 = THERMISTOR_CONSTANTS[1] * np.log(res)
        term3 = (
            THERMISTOR_CONSTANTS[2] * np.log(res) * np.log(res) * np.log(res)
        )
        tempK = 1.0 / (THERMISTOR_CONSTANTS[0] + term2 + term3)
        tempC = round((tempK - 273.15), 2)
        return tempC

    def test_connection(self):
        self.ser.write("T".encode("ascii"))
        line = self.ser.readline().decode("ascii").strip()
        if line is not None:
            return "OK"
        else:
            return "NC"

    def run(self):
        global experiment_flag
        if self.test_connection() == "OK":
            print("\nArduino connection established")
            if experiment_flag:

                self.set_dac_voltage(self.voltage)
                time.sleep(1)

                try:
                    print("\nTaking test temperature reading...")
                    temp, volts = self.read_temp()
                    if (temp < MAX_TEMP) and (temp > MIN_TEMP):
                        print("Test temperature reading good")
                        print("\nArduino logging started")
                    else:
                        print(
                            "\nTemperature out of range. Ending experiment."
                            "\nPlease check that the temperature controller"
                            "\nis powered on and re-start the experiment."
                        )
                        experiment_flag = False
                except Exception:
                    print(
                        "\nMath domain error. Temperature out of range."
                        "\nEnding experiment. Please check that the"
                        "\ntemperature controller is powered on and"
                        "\nre-start the experiment."
                    )
                    experiment_flag = False

                with open(self.temp_data_file_str, "w") as file:
                    writer = csv.writer(file, delimiter=",")
                    writer.writerow(["Datetime", "Temperature", "Volts"])

                while experiment_flag:
                    temp, volts = self.read_temp()
                    now = str(datetime.datetime.now())[:-4]
                    with open(self.temp_data_file_str, "a") as file:
                        writer = csv.writer(file, delimiter=",")
                        writer.writerow([now, temp, volts])
                    time.sleep(ADC_SAMPLE_PERIOD)
                print("\nArduino logging stopped")
                print("Setting voltage to 1550 (~20C)")
                self.set_dac_voltage(DEFAULT_VOLTAGE)
                self.close_connection()
            else:
                self.set_dac_voltage(self.voltage)
                self.close_connection()
        else:
            print("\nArduino connection failed")


# https://stackoverflow.com/a/44404713
class CamThread(QThread):

    # Signal function for sending pictures to main GUI display label
    changePixmap = pyqtSignal(QImage)

    def __init__(self, experiment_name, run_days, parent=None):
        super(CamThread, self).__init__(parent)
        self.runDays = run_days

    def run(self):
        global experiment_flag
        global brightness, contrast, gamma, sharpness, auto_exposure, exposure

        # If experiment is not running, preview camera feed
        if experiment_flag is False:
            try:
                cap = cv.VideoCapture(0)

                cap.set(cv.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

                while True:
                    cap.set(cv.CAP_PROP_BRIGHTNESS, brightness)
                    cap.set(cv.CAP_PROP_CONTRAST, contrast)
                    cap.set(cv.CAP_PROP_GAMMA, gamma)
                    cap.set(cv.CAP_PROP_SHARPNESS, sharpness)
                    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
                    cap.set(cv.CAP_PROP_EXPOSURE, exposure)

                    ret, frame = cap.read()
                    frame_copy = frame.copy()

                    if ret:
                        # https://stackoverflow.com/a/55468544/6622587
                        rgb_img = cv.cvtColor(frame_copy, cv.COLOR_BGR2RGB)
                        h, w, ch = rgb_img.shape
                        bytes_per_line = ch * w
                        qt_img = QImage(
                            rgb_img.data,
                            w,
                            h,
                            bytes_per_line,
                            QImage.Format_RGB888,
                        )
                        p = qt_img.scaled(1280, 720, Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)

                    if experiment_flag:
                        break

                cap.release()
                cv.destroyAllWindows()
            except Exception:
                print("\nCould not access camera")

        else:

            run_days = self.runDays
            experiment_dir = EXPERIMENTS_DIR + RIG_NAME + "/" + experiment_name
            videos_dir = experiment_dir + "/" + experiment_name + "_Videos/"
            video_frames_dir = (
                experiment_dir + "/" + experiment_name + "_VideoFrames/"
            )

            cap = cv.VideoCapture(0)

            cap.set(cv.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)
            cap.set(cv.CAP_PROP_BRIGHTNESS, brightness)
            cap.set(cv.CAP_PROP_CONTRAST, contrast)
            cap.set(cv.CAP_PROP_GAMMA, gamma)
            cap.set(cv.CAP_PROP_SHARPNESS, sharpness)
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
            cap.set(cv.CAP_PROP_EXPOSURE, exposure)

            print("\nOpened camera and set parameters")

            num_videos = int(run_days * (24 / VIDEO_HOURS))

            cnt_file_name = experiment_name + CNT_EXT
            cnt_file_str = experiment_dir + "/" + cnt_file_name

            with open(cnt_file_str, "w") as cnt_file:
                cnt_writer = csv.writer(cnt_file, delimiter=",")
                cnt_writer.writerow(cnt_header)
                print("Created contours file")

            experiment_timer = funcs.Timer(TIMER_PRECISION, quiet=True)
            experiment_timer.start()

            for i in range(num_videos):
                vid_num = i + 1
                frame_num = 1
                video_name = experiment_name + f"_video_{i+1}.avi"
                video_file_str = videos_dir + video_name

                video_writer = funcs.create_video_writer(
                    video_file_str, IMG_WIDTH, IMG_HEIGHT
                )
                print("Created video writer")

                frames_name = experiment_name + f"_frames_{i+1}.csv"
                frames_file_str = video_frames_dir + frames_name

                with open(frames_file_str, "w") as frames_file:
                    frames_writer = csv.writer(frames_file, delimiter=",")
                    frames_writer.writerow(
                        [
                            "VidNum",
                            "FrameNum",
                            "PCTime",
                            "SecFromVidStart",
                            "VidPlaybackTime",
                            "SecFromExperimentStart",
                        ]
                    )
                    print("\nCreated frames file for video", vid_num)

                return_flag, previous_img = cap.read()
                previous_img = cv.cvtColor(previous_img, cv.COLOR_BGR2GRAY)

                process_timer = funcs.Timer(TIMER_PRECISION, quiet=False)

                video_timer = funcs.Timer(TIMER_PRECISION, quiet=True)
                video_timer.start()

                while float(video_timer.elapsed()) < SEC_PER_VIDEO:
                    process_timer.start()
                    cnt_flag = False
                    return_flag, current_img = cap.read()
                    img_copy = current_img.copy()

                    current_img = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)

                    subtracted_img = funcs.subtract_images(
                        previous_img, current_img
                    )

                    _, binary_img = cv.threshold(
                        subtracted_img, BINARY_THRESH, 255, cv.THRESH_BINARY
                    )

                    contours, hierarchy = cv.findContours(
                        binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
                    )

                    saved_cnt_count = 0
                    for cnt in contours:
                        if saved_cnt_count > MAX_CONTOURS:
                            break
                        area = round(cv.contourArea(cnt), 0)
                        if area > CNT_AREA_THRESH:
                            saved_cnt_count += 1
                            cnt_flag = True
                            vid_time = round(frame_num / 30.0, 2)
                            x, y, w, h = cv.boundingRect(cnt)
                            cent_x = round(x + (w / 2), 0)
                            cent_y = round(y + (h / 2), 0)
                            perimeter = round(cv.arcLength(cnt, True), 2)
                            aspect = round(float(w) / h, 2)
                            rect_area = w * h
                            extent = round(float(area) / rect_area, 2)
                            pe_ex = round(perimeter / extent, 2)
                            ar_ex = round(area / extent, 2)
                            as_ex = round(aspect / extent, 2)

                            cv.rectangle(
                                img_copy,
                                (x, y),
                                (x + w, y + h),
                                (0, 255, 0),
                                1,
                            )

                            cnt_data = [
                                [
                                    vid_num,
                                    vid_time,
                                    frame_num,
                                    cent_x,
                                    cent_y,
                                    area,
                                    perimeter,
                                    aspect,
                                    extent,
                                    pe_ex,
                                    ar_ex,
                                    as_ex,
                                ]
                            ]

                            funcs.access_csv(cnt_file_str, cnt_data, "a")

                    # https://stackoverflow.com/a/55468544/6622587
                    if return_flag:
                        rgb_img = cv.cvtColor(img_copy, cv.COLOR_BGR2RGB)
                        h, w, ch = rgb_img.shape
                        bytes_per_line = ch * w
                        qt_img = QImage(
                            rgb_img.data,
                            w,
                            h,
                            bytes_per_line,
                            QImage.Format_RGB888,
                        )
                        p = qt_img.scaled(1280, 720, Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)

                    # Only check if at least one contour was detected per frame
                    # so that save only one frame
                    if cnt_flag:
                        bgr_img = cv.cvtColor(current_img, cv.COLOR_GRAY2BGR)
                        pc_time = str(datetime.datetime.now())
                        sec_from_vid_start = video_timer.elapsed()
                        vid_playback_time = frame_num * (1 / 30)
                        sec_from_experiment_start = experiment_timer.elapsed()

                        with open(frames_file_str, "a") as frames_file:
                            frames_writer = csv.writer(
                                frames_file, delimiter=","
                            )
                            frames_writer.writerow(
                                [
                                    vid_num,
                                    frame_num,
                                    pc_time,
                                    sec_from_vid_start,
                                    vid_playback_time,
                                    sec_from_experiment_start,
                                ]
                            )

                        video_writer.write(bgr_img)
                        frame_num += 1
                    process_timer.stop()
                    previous_img = current_img
                video_timer.stop()
                video_writer.release()
                cv.destroyAllWindows()
            experiment_timer.stop()
            cap.release()
            experiment_flag = False
            print("Experiment Complete")


class EggVidGetMetaGUI(QMainWindow, egg_vid_get_meta_gui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(EggVidGetMetaGUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Metadata Entry Form")

        self.tempProfileBox.addItems(temp_profiles)
        self.tempProfileBox.setCurrentText(DEFAULT_TEMPERATURE)
        self.tempProfileBox.currentIndexChanged.connect(self.set_temp_profile)
        self.set_temp_profile()

        self.saveMetadataButton.clicked.connect(self.set_metadata)

    def set_temp_profile(self):
        global dac_setting
        temp_profile = self.tempProfileBox.currentText()
        with open(MAIN_DIR + "temp_profiles/" + temp_profile + ".txt") as file:
            dac_setting = file.read().strip()
        print(f"\nTemp profile selected: {temp_profile}")
        print(f"Voltage: {dac_setting}")

    def set_metadata(self):
        global experiment_name
        global meta_vals_temp_list
        temp_profile = self.tempProfileBox.currentText()
        strain = self.strainLine.text()
        date = str(datetime.datetime.now().date())
        experiment_name = (
            strain + "_" + temp_profile + "_" + date + "_" + RIG_NAME
        )
        chip_num = self.chipLine.text()
        purpose = self.purposeLine.text()
        set_a_bacteria = self.setABacteriaLine.text()
        set_a_worms = self.setAWormsLine.text()
        set_b_bacteria = self.setBBacteriaLine.text()
        set_b_worms = self.setBWormsLine.text()
        try:
            set_a_worm_positions = [
                self.setATableWidget.item(r, 0).text() for r in range(16)
            ]
            set_b_worm_positions = [
                self.setBTableWidget.item(r, 0).text() for r in range(16)
            ]
            meta_vals_temp_list = [
                temp_profile,
                chip_num,
                RIG_NAME,
                purpose,
                set_a_bacteria,
                set_a_worms,
                set_b_bacteria,
                set_b_worms,
            ]
            meta_vals_temp_list.extend(set_a_worm_positions)
            meta_vals_temp_list.extend(set_b_worm_positions)
            print("\nSet metadata in temporary list")
        except AttributeError:
            print("\nFailed to set metadata. Please complete worm positions.")


class EggVidGetMainGUI(QMainWindow, egg_vid_get_main_gui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(EggVidGetMainGUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(f"egg-vid-get {VER_STR}")

        self.runtimeComboBox.addItems(runtime_options)
        self.runtimeComboBox.currentIndexChanged.connect(self.runtime_change)

        self.setMetadataButton.clicked.connect(self.open_meta_gui)

        self.brightnessSlider.setMinimum(-127)
        self.brightnessSlider.setMaximum(127)
        self.brightnessSlider.setValue(brightness)
        self.brightnessSlider.valueChanged.connect(self.brightness_change)
        self.brightnessValLabel.setText(str(self.brightnessSlider.value()))

        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(127)
        self.contrastSlider.setValue(contrast)
        self.contrastSlider.valueChanged.connect(self.contrast_change)
        self.contrastValLabel.setText(str(self.contrastSlider.value()))

        self.gammaSlider.setMinimum(16)
        self.gammaSlider.setMaximum(500)
        self.gammaSlider.setValue(gamma)
        self.gammaSlider.valueChanged.connect(self.gamma_change)
        self.gammaValLabel.setText(str(self.gammaSlider.value()))

        self.sharpnessSlider.setMinimum(0)
        self.sharpnessSlider.setMaximum(6)
        self.sharpnessSlider.setValue(sharpness)
        self.sharpnessSlider.valueChanged.connect(self.sharpness_change)
        self.sharpnessValLabel.setText(str(self.sharpnessSlider.value()))

        auto_exposure_options = [0, 1, 2, 3]
        auto_exposure_options = list(map(str, auto_exposure_options))
        self.autoExposureComboBox.addItems(auto_exposure_options)
        self.autoExposureComboBox.currentIndexChanged.connect(
            self.auto_exposure_change
        )
        self.autoExposureComboBox.setCurrentIndex(3)

        self.exposureSlider.setMinimum(1)
        self.exposureSlider.setMaximum(5000)
        self.exposureSlider.setValue(exposure)
        self.exposureSlider.valueChanged.connect(self.exposure_change)
        self.exposureValLabel.setText(str(self.exposureSlider.value()))

        self.startExperimentButton.clicked.connect(self.start_experiment)

        self.previewThread = CamThread("", 0)
        self.previewThread.changePixmap.connect(self.set_img)
        self.previewThread.start()

        # Test connection on startup. Won't start recording. Sends
        # default 20C value to Arduino, but this doesn't matter
        # yet since new connections to Arduino will soft reset it
        self.arduinoThread = ArduinoThread(dac_setting, "")
        self.arduinoThread.start()

    def brightness_change(self):
        global brightness
        slider_val = str(self.brightnessSlider.value())
        self.brightnessValLabel.setText(slider_val)
        brightness = int(slider_val)

    def contrast_change(self):
        global contrast
        slider_val = str(self.contrastSlider.value())
        self.contrastValLabel.setText(slider_val)
        contrast = int(slider_val)

    def gamma_change(self):
        global gamma
        slider_val = str(self.gammaSlider.value())
        self.gammaValLabel.setText(slider_val)
        gamma = int(slider_val)

    def sharpness_change(self):
        global sharpnes
        slider_val = str(self.sharpnessSlider.value())
        self.sharpnessValLabel.setText(slider_val)

    def auto_exposure_change(self):
        global auto_exposure
        auto_exposure = int(self.autoExposureComboBox.currentText())

    def exposure_change(self):
        global exposure
        slider_val = str(self.exposureSlider.value())
        self.exposureValLabel.setText(slider_val)
        exposure = int(slider_val)

    def open_meta_gui(self):
        self.meta_gui = EggVidGetMetaGUI()
        self.meta_gui.show()

    def set_img(self, img):
        self.frameDisplayLabel.setPixmap(QPixmap.fromImage(img))

    def runtime_change(self):
        runtime = self.runtimeComboBox.currentText()
        print(f"\nRuntime: {runtime}")

    def start_experiment(self):
        experiment_dir = EXPERIMENTS_DIR + RIG_NAME + "/" + experiment_name
        videos_dir = experiment_dir + "/" + experiment_name + "_Videos/"
        video_frames_dir = (
            experiment_dir + "/" + experiment_name + "_VideoFrames/"
        )

        try:
            os.mkdir(experiment_dir)
            os.mkdir(videos_dir)
            os.mkdir(video_frames_dir)
            print("\nCreated experiment directories")
            print("\nBeginning experiment...")

            global experiment_flag
            experiment_flag = True
            time.sleep(3)
            runtime = self.runtimeComboBox.currentText()
            run_days = int(runtime[0])
            now = datetime.datetime.now()
            start_date = str(now)[:10]
            start_time = str(now)[11:19]
            end_datetime = str(now + datetime.timedelta(hours=24 * run_days))
            end_date = str(end_datetime)[:10]
            meta_vals_to_add = [
                experiment_name,
                start_date,
                start_time,
                runtime,
                end_date,
            ]
            meta_vals = meta_vals_to_add + meta_vals_temp_list
            meta_name = experiment_name + "_metadata.csv"
            meta_file_str = experiment_dir + "/" + meta_name

            with open(meta_file_str, "w") as meta_file:
                meta_writer = csv.writer(meta_file, delimiter=",")
                meta_writer.writerow(meta_fields)
                meta_writer.writerow(meta_vals)
            print("\nSaved metadata")

            temp_data_file_name = experiment_name + "_temp_data.csv"
            temp_data_file_str = experiment_dir + "/" + temp_data_file_name

            self.startDisplayLabel.setText(str(now)[:19])
            self.endDisplayLabel.setText(str(end_datetime)[:19])

            print("\nStarting Arduino logging...")

            self.arduinoThread = ArduinoThread(dac_setting, temp_data_file_str)
            self.arduinoThread.start()
            # Approximate time for Arduino thread to start, set temp,
            # write temperature file, and begin recording
            time.sleep(4.037)

            self.processThread = CamThread(experiment_name, run_days)
            self.processThread.changePixmap.connect(self.set_img)
            self.processThread.start()
        except FileExistsError:
            print(
                "\nExperiment already exists. Please delete the folder"
                '\nfrom the last attempt then click "Start Experiment"'
                "\nagain."
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_gui = EggVidGetMainGUI()
    main_gui.show()
    sys.exit(app.exec_())
