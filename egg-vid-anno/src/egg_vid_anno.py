import os
import csv
import cv2
import sys
import math
import egg_vid_anno_gui
import pandas as pd
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileSystemModel,
                             QTreeView)


VER_STR = '3.2'

global vid_dir
vid_dir = ''
global cnt_file_path
cnt_file_path = ''
global anno_file_path
anno_file_path = ''
global experiment_dir
experiment_dir = ''
global experiment_name
experiment_name = ''
global last_anno_vid
last_anno_vid = 0
global cnt_file_index
cnt_file_index = 0
global experiment_selected_flag
experiment_selected_flag = False

anno_file_header = ['ContourIndex',
                    'VidNum',
                    'VidTime',
                    'FrameNum',
                    'X',
                    'Y',
                    'Area',
                    'Perimeter',
                    'Aspect',
                    'Extent',
                    'Pe/Ex',
                    'Ar/Ex',
                    'As/Ex',
                    'LaneNum',
                    'PCTime',
                    'SecFromVidStart',
                    'SecFromExperimentStart',
                    'Object',
                    'New']


def access_csv(file_str, data, usage_str):
    with open(file_str, usage_str, newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for el in data:
            writer.writerow(el)


class ProcessThread(QThread):

    def __init__(self, start_video, start_cnt_file_index,
                 start_anno_file_index, parent=None):
        super(ProcessThread, self).__init__(parent)
        self.startVideo = start_video

        self.startCntFileIndex = start_cnt_file_index
        self.startAnnoFileIndex = start_anno_file_index

    def run(self):
        start_vid_num = int(self.startVideo)
        start_cnt_index = int(self.startCntFileIndex) - 2
        start_anno_index = int(self.startAnnoFileIndex) - 2
        last_vid_num = start_vid_num

        vid_path = (vid_dir + experiment_name +
                    '_video_{}.avi'.format(start_vid_num))

        vid = cv2.VideoCapture(vid_path)
        print('\nVideo {} loaded'.format(start_vid_num))

        cnt_data = pd.read_csv(cnt_file_path, header=0)
        print('\nContour data loaded')
        # cnt_data = cnt_data.dropna()

        cnt_index = start_cnt_index
        anno_index = start_anno_index
        anno_flag = True

        while True:
            write_flag = False

            try:
                vid_num = cnt_data.iloc[cnt_index]['VidNum']
            except IndexError:
                print('\nThis is the end of the last video')
                print('\nAre you done annotating?')
                key = input('y/n: ')
                if key == 'y':
                    break
                else:
                    cnt_index -= 1
                    frame_num = int(cnt_data.iloc[cnt_index]['FrameNum'])
                    print('\nGoing back to last frame')
                    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 2)
                    ret, img = vid.read()
                    print('\n', cnt_index+2)

            if vid_num != last_vid_num:
                vid.release()
                print('\nVideo {} released'.format(int(last_vid_num)))
                print('\nSwitching to video', int(vid_num))
                vid_path = (vid_dir + experiment_name +
                            '_video_{}.avi'.format(int(vid_num)))
                vid = cv2.VideoCapture(vid_path)
                print('\nVideo {} loaded'.format(int(vid_num)))
                last_vid_num = vid_num
                anno_flag = True

            cent_x = cnt_data.iloc[cnt_index]['X']
            cent_y = cnt_data.iloc[cnt_index]['Y']
            frame_num = int(cnt_data.iloc[cnt_index]['FrameNum'])

            vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            ret, img = vid.read()

            if img is None:
                print('\nFailed to read frame')
                print('\nSomething is likely wrong with the video file')
                print('\nLast video:', vid_num)
                print('Last frame:', frame_num)
                print('\nExiting annotation state')
                break

            img_copy = img.copy()

            last_vid_num = cnt_data.iloc[cnt_index]['VidNum']
            vid_time = cnt_data.iloc[cnt_index]['VidTime']
            area = cnt_data.iloc[cnt_index]['Area']
            perimeter = cnt_data.iloc[cnt_index]['Perimeter']
            aspect = cnt_data.iloc[cnt_index]['Aspect']
            ex = cnt_data.iloc[cnt_index]['Extent']
            pe_ex = cnt_data.iloc[cnt_index]['Pe/Ex']
            ar_ex = cnt_data.iloc[cnt_index]['Ar/Ex']
            as_ex = cnt_data.iloc[cnt_index]['As/Ex']
            lane_num = cnt_data.iloc[cnt_index]['LaneNum']
            pc_time = cnt_data.iloc[cnt_index]['PCTime']
            sec_from_vid_start = cnt_data.iloc[cnt_index]['SecFromVidStart']
            sec_from_exp_start = cnt_data.iloc[
                                               cnt_index][
                                               'SecFromExperimentStart']

            h = int(math.sqrt(area / (ex * aspect)))
            w = int(aspect * h)
            x = int(cent_x - (w / 2))
            y = int(cent_y - (h / 2))

            try:
                last_frame_num = int(cnt_data.iloc[cnt_index-1]['FrameNum'])
            except Exception:
                last_frame_num = 0

            if frame_num == last_frame_num:
                cv2.rectangle(img_copy, (x-5, y-5), (x+w+5, y+h+5),
                              (0, 255, 0), 1)
            else:
                cv2.rectangle(img_copy, (x-5, y-5), (x+w+5, y+h+5),
                              (0, 255, 255), 1)

            cv2.imshow('Current Image', img_copy)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

            obj = ''
            new_obj = ''
            if anno_flag:
                key = input('\nIndex {}: '.format(cnt_index+2))
            else:
                key = input()
            if ' ' in key:
                new_obj = 'yes'
            else:
                new_obj = 'no'
            if 'o' in key:
                overwrite = 'yes'
            else:
                overwrite = 'no'
            if 't' in key:
                obj = 'egg'
                print('\nIndex', cnt_index+2)
                print('   New:', new_obj)
                print('   Object:', obj)
                print('   Overwrite:', overwrite)
                write_flag = True
                anno_flag = True
                cnt_index += 1
            elif 'q' in key:
                num_eggs = input('\nEggs in cluster: ')
                try:
                    num_eggs = int(num_eggs)
                    obj = str(num_eggs) + ' eggs'
                    print('\nIndex', cnt_index+2)
                    print('   New:', new_obj)
                    print('   Object:', obj)
                    print('   Overwrite:', overwrite)
                    write_flag = True
                    anno_flag = True
                    cnt_index += 1
                except Exception:
                    print("\nThat is not an integer, please enter 'q' "
                          "and try again")
            elif 'w' in key:
                obj = '2 eggs'
                print('\nIndex', cnt_index+2)
                print('   New:', new_obj)
                print('   Object:', obj)
                print('   Overwrite:', overwrite)
                write_flag = True
                anno_flag = True
                cnt_index += 1
            elif 'y' in key:
                obj = 'l1'
                print('\nIndex', cnt_index+2)
                print('   New:', new_obj)
                print('   Object:', obj)
                print('   Overwrite:', overwrite)
                write_flag = True
                anno_flag = True
                cnt_index += 1
            elif 'e' in key:
                obj = 'adult'
                print('\nIndex', cnt_index+2)
                print('   New:', new_obj)
                print('   Object:', obj)
                print('   Overwrite:', overwrite)
                write_flag = True
                anno_flag = True
                cnt_index += 1
            elif 'r' in key:
                obj = 'junk'
                print('\nIndex', cnt_index+2)
                print('   New:', new_obj)
                print('   Object:', obj)
                print('   Overwrite:', overwrite)
                write_flag = True
                anno_flag = True
                cnt_index += 1
            elif 'a' in key:
                cnt_index -= 1
                print('\n', cnt_index+2)
                anno_flag = False
            elif 'x' in key:
                break
            else:
                cnt_index += 1
                print('\n', cnt_index+2)
                anno_flag = False

            out_data = [cnt_index+1, vid_num, vid_time, frame_num,
                        cent_x, cent_y, area, perimeter, aspect, ex, pe_ex,
                        ar_ex, as_ex, lane_num, pc_time, sec_from_vid_start,
                        sec_from_exp_start, obj, new_obj]

            if overwrite == 'yes':
                write_flag = False
                anno_data = pd.read_csv(anno_file_path, header=0)
                cnt_anno_check_df = anno_data[anno_data['ContourIndex']
                                              == cnt_index+1]
                if cnt_anno_check_df.empty:
                    print("\nThis contour has not been annotated")
                    cnt_index -= 1
                else:
                    anno_index_list = cnt_anno_check_df.index.tolist()
                    anno_index_match = int(anno_index_list[0])
                    anno_data.iloc[anno_index_match] = out_data
                    anno_data.to_csv(anno_file_path, index=False)
                    print("\nAnnotation overwritten")

            if write_flag:
                anno_data = pd.read_csv(anno_file_path, header=0)
                cnt_anno_check_df = anno_data[anno_data['ContourIndex']
                                              == cnt_index+1]
                if cnt_anno_check_df.empty:
                    access_csv(anno_file_path, [out_data], 'a')
                    anno_index += 1
                    print("\nAnnotation saved")
                else:
                    print("\nThis contour is already annotated as:")
                    new_val = cnt_anno_check_df['New'].iloc[0]
                    obj_val = cnt_anno_check_df['Object'].iloc[0]
                    print("New:", new_val)
                    print("Object:", obj_val)
                    cnt_index -= 1

        cv2.destroyAllWindows()
        vid.release()


class MainGUI(QMainWindow, egg_vid_anno_gui.Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainGUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(f'egg-vid-anno v{VER_STR}')

        self.startAnnoButton.clicked.connect(self.start_annotating)
        self.createAnnoFileButton.clicked.connect(self.create_anno_file)
        self.selectContoursFileButton.clicked.connect(
                                         self.select_contours_file
                                         )
        self.getLastAnnoButton.clicked.connect(self.get_last_annotation)

    def create_anno_file(self):
        if os.path.exists(anno_file_path):
            print('\nThe annotations file for this experiment already exists')
        else:
            try:
                access_csv(anno_file_path, [anno_file_header], 'w')
                print('\nCreated annotations file')
                self.annoFileCreatedLabel.setText('Done')
                self.get_last_annotation()
            except Exception:
                print('\nPlease select an experiment directory')

    def get_directory_strings(self, signal):
        global vid_dir
        global cnt_file_path
        global anno_file_path
        global experiment_dir
        global experiment_name
        global experiment_selected_flag

        cnt_file_path = self.model.filePath(signal)

        cnt_file_name = cnt_file_path[cnt_file_path.rfind('/') + 1:]

        experiment_dir = cnt_file_path[:cnt_file_path.rfind('/')]

        experiment_name = experiment_dir[experiment_dir.rfind('/') + 1:]

        vid_dir = f'{experiment_dir}/{experiment_name}_Videos/'
        anno_file_path = f'{experiment_dir}/{experiment_name}_annotations.csv'

        if os.path.exists(anno_file_path):
            self.annoFileCreatedLabel.setText('Done')

        print('\nContours file selected:')
        print(cnt_file_name)

        print('\nExperiment:')
        print(experiment_name)

        print('\nVideo directory:')
        print(vid_dir)

        experiment_selected_flag = True

    def get_last_annotation(self):
        global last_anno_vid
        global cnt_file_index

        try:
            cnt_data = pd.read_csv(cnt_file_path, header=0)
            anno_data = pd.read_csv(anno_file_path, header=0)
            if anno_data.empty:
                print('\nThis experiment has no annotations')
                print('\nSetting values to the beginning')
                last_anno_vid = 1
                cnt_file_index = 2
                anno_file_index = 2
                self.vidNumLineEdit.setText(str(last_anno_vid))
                self.cntFileIndexLineEdit.setText(str(cnt_file_index))
                self.annoFileIndexLineEdit.setText(str(anno_file_index))
            else:
                last_anno_vid = int(anno_data.iloc[-1]['VidNum'])
                last_anno_frame = int(anno_data.iloc[-1]['FrameNum'])
                cnt_file_index_list = (cnt_data.index[(cnt_data['FrameNum']
                                                      == last_anno_frame) &
                                                      (cnt_data['VidNum'] ==
                                                      last_anno_vid)].tolist())
                cnt_file_index = int(cnt_file_index_list[0]) + 2
                anno_file_index_vals = anno_data.index.tolist()
                anno_file_index = int(anno_file_index_vals[-1] + 2)
                print('\nLast Video Number:', last_anno_vid)
                print('Last Frame Number:', last_anno_frame)
                print('Last Contour File Index:', cnt_file_index)
                print('Last Annotations File Index:', anno_file_index)
                self.vidNumLineEdit.setText(str(last_anno_vid))
                self.cntFileIndexLineEdit.setText(str(cnt_file_index))
                self.annoFileIndexLineEdit.setText(str(anno_file_index))
        except Exception:
            print('\nPlease select an experiment directory')

    def select_contours_file(self):
        self.model = QFileSystemModel()
        self.model.setRootPath('/')
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setAnimated(False)
        self.tree.setIndentation(10)
        self.tree.setSortingEnabled(True)
        self.tree.setWindowTitle("Directory View")
        self.tree.resize(640, 480)
        self.tree.setColumnWidth(0, 300)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.doubleClicked.connect(self.get_directory_strings)
        print('\nSelecting contours file...')
        self.tree.show()

    def start_annotating(self):
        if experiment_selected_flag:
            start_vid = self.vidNumLineEdit.text()
            start_cnt_file_index = self.cntFileIndexLineEdit.text()
            start_anno_file_index = self.annoFileIndexLineEdit.text()
            print('\nStarting at Video:', start_vid)
            print('Starting at Contour File Index:', start_cnt_file_index)
            print('Starting at Annotation File Index:', start_anno_file_index)
            self.processThread = ProcessThread(start_vid, start_cnt_file_index,
                                               start_anno_file_index)
            self.processThread.start()
        else:
            print('\nPlease select an experiment directory')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = MainGUI()
    main_gui.show()
    sys.exit(app.exec_())
