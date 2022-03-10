"""
This program is a processing tool to be used on the contour data
from the temperature controlled egg counting devices after experimental
data collection is complete.

The tool allows the user to select the main directory of a completed
experiment, draw boxes over the lanes of the microfluidic chip using
a frame from the first experimental video, then assign to each contour
which lane the contour was detected in, according to the experimenter's
lane drawings. The tool also adds timing information to the main contours
file from the video frames files collected during the experiment.

Additionally, the tool allows the user to remove lanes that are deemed
un-useable for the final data analysis of the experiment.

Author: Cody Jarrett
Organization: Phillips Lab, University of Oregon
"""

import os
import sys
import warnings
import cv2 as cv
import pandas as pd
import egg_lane_proc_gui
import egg_lane_proc_functions as funcs
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileSystemModel,
                             QTreeView)


VER_STR = 'v1.2'

global sel_img

lanes_header = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

# For drawing lane boxes
x1, y1 = 0, 0
x2, y2 = 0, 0
x3, y3 = 0, 0
x4, y4 = 0, 0
pt_1_flag = False
pt_2_flag = False
pt_3_flag = False
pt_4_flag = False
single_box_points = []
all_boxes_points = []


def draw_box(event, x, y, flags, param):
    """
    Function that allows the user to draw a box on an image by clicking four
    points. Upon clicking the fourth point, the third and fourth points are
    automatically connected with a line, and the user may move on to draw
    another box.

    Args
    ----
    event : cv button event
        Tracks whether mouse has been clicked
    x : int
        x-coordinate of the point where the user clicks
    y : int
        y-coordinate of the point where the user clicks

    Returns
    -------
    None
    """

    global x1, y1, x2, y2, x3, y3, x4, y4
    global pt_1_flag, pt_2_flag, pt_3_flag, pt_4_flag
    global single_box_points, all_boxes_points
    if event == cv.EVENT_LBUTTONDOWN:
        if pt_1_flag is False:
            pt_1_flag = True
            x1, y1 = x, y
            single_box_points.extend([x1, y1])
        elif pt_2_flag is False:
            pt_2_flag = True
            x2, y2 = x, y
            cv.line(sel_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            single_box_points.extend([x2, y2])
        elif pt_3_flag is False:
            pt_3_flag = True
            x3, y3 = x, y
            cv.line(sel_img, (x2, y2), (x3, y3), (0, 0, 255), 1)
            single_box_points.extend([x3, y3])
        elif pt_4_flag is False:
            x4, y4 = x, y
            cv.line(sel_img, (x3, y3), (x4, y4), (0, 0, 255), 1)
            cv.line(sel_img, (x4, y4), (x1, y1), (0, 0, 255), 1)
            single_box_points.extend([x4, y4])
            all_boxes_points.append(single_box_points)
            single_box_points = []
            pt_1_flag = False
            pt_2_flag = False
            pt_3_flag = False


class EggLaneProcGUI(QMainWindow, egg_lane_proc_gui.Ui_MainWindow):
    """
    A class to generate and use the main GUI of the egg lane
    processing tool

    Attributes
    ----------
    exp_dir : str
        a formatted string track the directory of the
        selected experiment
    exp_name : str
        a formatted string to track the name of the
        selected experiment
    lanes_csv_str : str
        a formatted string to track the path of the lanes
        csv file
    lanes_pic_str : str
        a formatted string to track the path of the lanes
        picture with boxes drawn on it
    vid_dir : str
        a formatted string to track the path of the video
        used for drawing the lane boxes
    vid_frames_dir : str
        a formatted string to track the directory of the
        video frames files
    first_vid_str : str
        a formatted string to track the path of the first
        video of the experiment
    contours_file_path : str
        a formatted string to track the path of the contours file
    contours : pandas.DataFrame
        a DataFrame to store the contours information from the
        experiment's contours csv file
    data_loaded_flag: boolean
        a flag to track whether the contour data has been loaded
    """

    def __init__(self, parent=None):
        """
        Parameters
        ----------
        None

        Raises
        ------
        None
        """

        super(EggLaneProcGUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(f"egg-lane-proc {VER_STR}")

        # Buttons and methods for "Draw Lanes" section
        self.selectExperimentButton.clicked.connect(self.select_experiment)
        self.drawLanesButton.clicked.connect(self.draw_lanes)
        self.assignLanesButton.clicked.connect(self.assign_lanes_and_times)

        # Buttons and methods for "Remove Lanes" section
        self.loadContoursFileButton.clicked.connect(self.load_contours_file)
        self.removeLanesButton.clicked.connect(self.remove_lanes)

        # Flag to track whether data is loaded
        self.data_loaded_flag = False

        # Flag to track whether experiment is selected
        self.selected_flag = False

        self.firstVidSecToSkipLineEdit.setText('0')

    def select_experiment(self):
        """
        Defines and displays a file browser window for
        selecting which experiment the tool will be used
        to process.

        Parameters
        ----------
        None

        Raises
        ------
        None
        """

        self.model = QFileSystemModel()
        self.model.setRootPath('/')
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setAnimated(False)
        self.tree.setIndentation(10)
        self.tree.setSortingEnabled(True)
        self.tree.setWindowTitle("Dir View")
        self.tree.resize(640, 480)
        self.tree.setColumnWidth(0, 300)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.doubleClicked.connect(self.get_experiment_dir)
        print('\nSelecting experiment directory...')
        self.tree.show()

    def get_experiment_dir(self, signal):
        """
        Mouse callback that allows the user to select an experiment
        by double-clicking on it in a directory browser, then stores various
        directory and file strings for the selected experiment

        Parameters
        ----------
        signal : str
            The string that the user double-clicked in the
            file browser window defined by select_experiment()

        Raises
        ------
        None
        """

        self.exp_dir = f'{self.model.filePath(signal)}/'
        temp_str = self.exp_dir[:self.exp_dir.rfind('/')]
        self.exp_name = temp_str[temp_str.rfind('/') + 1:]

        self.selectedExperimentLabel.setText(self.exp_name)

        self.lanes_csv_str = f'{self.exp_dir}{self.exp_name}_lanes.csv'
        self.lanes_pic_str = f'{self.exp_dir}{self.exp_name}_lanes_pic.jpg'
        self.vid_dir = f'{self.exp_dir}{self.exp_name}_Videos/'
        self.vid_frames_dir = f'{self.exp_dir}{self.exp_name}_VideoFrames/'
        self.first_vid_str = f'{self.vid_dir}{self.exp_name}_video_1.avi'

        print('\nExperiment directory selected:')
        print('\n', self.exp_dir)

    def draw_lanes(self):
        """
        Button callback that allows the user to draw boxes over
        the egg counter chip lanes in a sample frame from one of
        the experiment videos

        Parameters
        ----------
        None

        Raises
        ------
        Exception
            If could not save lane boxes image, create lanes csv file,
            or write to lanes csv file
        """

        global all_boxes_points, sel_img

        try:
            if os.path.isfile(self.lanes_csv_str):
                print('\nLanes have already been selected.')
                print('Please delete old selection files then try again.')
            else:
                print('\nLoading video frame...')

                first_vid = cv.VideoCapture(self.first_vid_str)

                first_vid_sec_ignore = self.firstVidSecToSkipLineEdit.text()
                frame_ignore_num = 30 * int(first_vid_sec_ignore)

                # Skip ahead to get a good image for drawing on
                for i in range(frame_ignore_num):
                    ret, sel_img = first_vid.read()

                cv.namedWindow('Selection Image')
                cv.setMouseCallback('Selection Image', draw_box)

                try:
                    while 1:
                        cv.imshow('Selection Image', sel_img)

                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break

                        if cv.waitKey(1) == 13:
                            try:
                                write_ret = cv.imwrite(self.lanes_pic_str,
                                                       sel_img)
                            except Exception:
                                print('\nProblem trying to save lane boxes '
                                      'image')
                            if write_ret:
                                print('\nSaved lane boxes image')
                            else:
                                print('\nCould not save lane boxes image')

                            try:
                                funcs.access_csv(self.lanes_csv_str,
                                                 [lanes_header],
                                                 'w')
                                funcs.access_csv(self.lanes_csv_str,
                                                 all_boxes_points,
                                                 'a')
                                print('Wrote lane box coordinates CSV file')
                            except Exception:
                                print('\nCould not write lane box coordinates '
                                      'CSV file')

                            break

                    first_vid.release()
                    cv.destroyAllWindows()
                    print('\nCleaned up lane drawing instance')

                except cv.error:
                    cv.destroyAllWindows()
                    print('\nThe video frame could not be read. Make sure'
                          '\nthe number of seconds to skip in the first video'
                          '\nis less than the total length of the video.')

        except AttributeError:
            print('\nExperiment has not been selected. Please select an '
                  'experiment and try again.')

    def assign_lanes_and_times(self):
        """
        Button callback that adds lane numbers from lane selections
        and timing info from video frames files to the experiment's main
        contours file.

        Parameters
        ----------
        None

        Raises
        ------
        AttributeError
            Informs the user an experiment has not been selected if
            the lanes_csv_str, exp_dir or exp_name attributes have
            no value

        Exception
            If could not load contours file
        """
        try:
            lanes = pd.read_csv(self.lanes_csv_str, header=0)
            print('\nLoaded lane boxes')

            self.contours_file_path = (f'{self.exp_dir}/{self.exp_name}'
                                       '_contours.csv')
            self.selected_flag = True
        except AttributeError:
            print('\nExperiment has not been selected. Please select an '
                  'experiment and try again.')

        if self.selected_flag:
            # Load contours file
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    print('\nLoading contour data...')
                    self.contours = pd.read_csv(self.contours_file_path)
                    print('Data loaded')
            except Exception:
                print('\nCould not load data. Did you select an experiment?')

            if 'PCTime' in self.contours.columns:
                if self.overwriteLanesCheckBox.isChecked():
                    # Assign lanes
                    print('\nOverwriting lane assignments...')
                    lane_nums = [0]*len(self.contours)
                    for i, row in enumerate(self.contours.itertuples()):
                        x = row.X
                        for l_num in range(len(lanes)):
                            if funcs.is_obj_in_lane(x, lanes, l_num):
                                lane_num = l_num + 1
                                lane_nums[i] = lane_num

                    self.contours['LaneNum'] = lane_nums

                    self.contours.to_csv(self.contours_file_path, index=False)

                    print('Lane assignments overwritten')
                else:
                    print('\nLanes and timing have already been assigned and '
                          '\nthe overwrite lanes option was not selected.'
                          '\nNo changes were made')

            else:
                # Assign lanes
                print('\nAssigning lanes and timing info...')
                lane_nums = [0]*len(self.contours)
                for i, row in enumerate(self.contours.itertuples()):
                    x = row.X
                    for l_num in range(len(lanes)):
                        if funcs.is_obj_in_lane(x, lanes, l_num):
                            lane_num = l_num + 1
                            lane_nums[i] = lane_num

                self.contours['LaneNum'] = lane_nums

                vid_frame_dfs = []
                for filename in os.listdir(self.vid_frames_dir):
                    file_path = self.vid_frames_dir + filename
                    df = pd.read_csv(file_path)
                    diff = df['FrameNum'][0] - 1
                    df['FrameNum'] = [(x - diff) for x in df['FrameNum']]
                    vid_frame_dfs.append(df)

                vid_frames = pd.concat(vid_frame_dfs, ignore_index=True)

                new_contours_df = pd.merge(self.contours,
                                           vid_frames,
                                           on=['VidNum', 'FrameNum'])

                # Delete VidPlaybackTime column because it has the same info
                # as the VidTime column
                del new_contours_df['VidPlaybackTime']

                new_contours_df.to_csv(self.contours_file_path, index=False)

                print('Lanes assigned')

    def load_contours_file(self):
        """
        Button callback that loads the main contours file
        for removing specific lanes.

        Parameters
        ----------
        None

        Raises
        ------
        AttributeError
            If the experiment has not been selected, and therefore
            the self.contours_file_path attribute has not been
            assigned

        Exception
            If could not load contour data or could not determine
            lanes with highest number of contours
        """

        try:
            self.contours_file_path = (f'{self.exp_dir}/{self.exp_name}'
                                       '_contours.csv')
            self.selected_flag = True
        except AttributeError:
            print('\nExperiment has not been selected. Please select an '
                  'experiment and try again.')

        if self.selected_flag:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    print('\nLoading data...')
                    self.contours = pd.read_csv(self.contours_file_path)
                    self.numOriginalContoursValueLabel.setText(
                                        f'{len(self.contours)+1:,}'
                                        )
                    print('\nData loaded')
                    print('Contours counted')
                    self.data_loaded_flag = True
            except Exception:
                print('\nCould not load data')

            try:
                lanes_list = self.contours['LaneNum'].unique()

                lanes_with_most_contours = []
                for l_num in lanes_list:
                    lane_df = self.contours[self.contours['LaneNum'] == l_num]
                    num_contours = len(lane_df)
                    if num_contours > 5000:
                        lanes_with_most_contours.append([l_num, num_contours])

                # If lanes_with_most_contours is empty
                if not lanes_with_most_contours:
                    print('\nNo lanes contain more than 5000 contours')
                else:
                    print('\nThe following lanes have more than '
                          '5000 contours:')
                    for el in sorted(lanes_with_most_contours,
                                     # by num_contours
                                     key=lambda lane: lane[1],
                                     # descending
                                     reverse=True):
                        l_num, num_contours = el
                        print('{}: {:,}'.format(l_num, num_contours))

            except Exception:
                print('\nCould not determine highest contour lanes')

    def remove_lanes(self):
        """
        Button callback that removes lanes specified by the user
        that have been deemed un-useable for experiment data analysis.

        Parameters
        ----------
        None

        Raises
        ------
        AttributeError
            If user has not selected an experiment, data_loaded_flag will
            not be assigned

        ValueError
            If user leaves the lanes to remove field blank, misses a comma, or
            enters a character that is not a number
        """
        if self.data_loaded_flag:
            lanes_list = self.contours['LaneNum'].unique()

            lanes_to_remove_str = self.lanesToRemoveLineEdit.text()
            lanes_to_remove_list = lanes_to_remove_str.split(',')

            invalid_lane_num_flag = False

            try:
                lanes_to_remove_list = [int(x) for x in lanes_to_remove_list]

                lanes_not_in_file = []
                for num in lanes_to_remove_list:
                    if num not in lanes_list:
                        invalid_lane_num_flag = True
                        lanes_not_in_file.append(num)

                if invalid_lane_num_flag:
                    print('\nThe following lanes are not in the contours file:'
                          '\n\n{}'
                          '\n\nPlease remove these lanes from the list '
                          'and try again'.format(lanes_not_in_file))
                else:
                    lanes_to_keep_list = [x for x in lanes_list if
                                          x not in lanes_to_remove_list]

                    print('\nProcessing...')

                    lane_dfs_to_keep = []
                    for l_num in lanes_to_keep_list:
                        lane_to_keep_df = self.contours[
                                        self.contours['LaneNum'] == l_num
                                        ]
                        lane_dfs_to_keep.append(lane_to_keep_df)

                    keeps_df = pd.concat(lane_dfs_to_keep, ignore_index=True)

                    keeps_df.sort_values(by=['VidNum', 'VidTime'],
                                         inplace=True)
                    keeps_df = keeps_df.reset_index(drop=True)

                    # Remove leading, trailing, and multiple spaces...
                    cleaned_lanes_str = " ".join(lanes_to_remove_str.split())
                    # then replace single spaces with no space
                    cleaned_lanes_str = cleaned_lanes_str.replace(" ", "")

                    out_file_str = (f'{self.exp_dir}/{self.exp_name}'
                                    f'_contours_removed_'
                                    f'{cleaned_lanes_str}.csv')

                    keeps_df.to_csv(out_file_str, index=False)
                    print('Done')

                    self.numContoursAfterLaneRemoveValueLabel.setText(
                                                    f'{len(keeps_df)+1:,}'
                                                    )

            except ValueError:
                print('\nCould not accept input for lanes to remove. '
                      '\nYou left the field blank, missed a comma, or '
                      '\nthere is a character that is not a number.')

        else:
            print('\nCannot remove lanes. The contours file has '
                  'not been loaded. \nPlease double-check that '
                  'the experiment is selected and \nyou have '
                  'clicked the "Load Contours File" button.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = EggLaneProcGUI()
    main_gui.show()
    sys.exit(app.exec_())
