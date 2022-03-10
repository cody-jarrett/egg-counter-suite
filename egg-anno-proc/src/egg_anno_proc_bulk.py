import os
import sys
import shutil
import numpy as np
import pandas as pd
import egg_ana_gui
import egg_ana_functions as funcs
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileSystemModel,
                             QTreeView)


VER_STR = '1.0'


work_from_home = True

if work_from_home:
    ACT_EXP_DIR = 'E:/Work/Dropbox (CITP)/Egg Counter/Experiments/'
else:
    ACT_EXP_DIR = 'C:/Users/Coruscant/Dropbox (CITP)/Egg Counter/Experiments/'
# ACT_EXP_DIR = 'D:/MyStuff/Desktop/BulkAnaDir/'
REANALYSIS_FLAG = False


class EggAnaGUI(QMainWindow, egg_ana_gui.Ui_MainWindow):

    def __init__(self, parent=None):

        super(EggAnaGUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(f"bulk-egg-ana v{VER_STR}")

        self.anaButton.clicked.connect(self.analyze)

        self.minEggsLineEdit.setText('10')

    def _set_paths(self, exp_dir):
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

        self.exp_dir = f'{exp_dir}/'
        temp_str = self.exp_dir[:self.exp_dir.rfind('/')]
        self.exp_name = temp_str[temp_str.rfind('/') + 1:]

        self.anno_data_path = f'{self.exp_dir}{self.exp_name}_annotations.csv'

        self.ana_dir = f'{self.exp_dir}{self.exp_name}_Analysis/'

        self.egg_graphs_dir = f'{self.ana_dir}EggGraphs/'
        self.egg_hist_dir = f'{self.egg_graphs_dir}Histograms/'
        self.egg_spacing_dir = f'{self.egg_graphs_dir}IntervalSpacings/'

        self.t_dir = f'{self.ana_dir}Temperature/'
        self.t_data_path = f'{self.exp_dir}{self.exp_name}_temp_data.csv'
        self.t_summary_csv_str = (f'{self.t_dir}{self.exp_name} '
                                  f'Temperature Stats.csv')

        print('\nExperiment directory selected:')
        print('\n', self.exp_dir)

    def _plot_eggs(self, dfs):
        funcs.plot_all_lanes_raw(dfs,
                                 self.min_obj_num,
                                 save=True,
                                 save_dir=self.egg_graphs_dir,
                                 title_str=f'Raw Eggs '
                                           f'{self.exp_name}')

        funcs.plot_all_lanes_intervals(dfs,
                                       self.min_obj_num,
                                       save=True,
                                       save_dir=self.egg_graphs_dir,
                                       title_str=(f'Egg Intervals '
                                                  f'{self.exp_name}'))

        for i, df in enumerate(dfs):
            if not df.empty:
                num_eggs = len(df)
                hist_title = (f'Lane {i+1} Log Interval Histogram '
                              f'({num_eggs} Eggs)')
                spacing_title = (f'Lane {i+1} Raw Egg Interval Spacing '
                                 f'({num_eggs} Eggs)')
                funcs.plot_single_lane_info(df,
                                            i,
                                            save=True,
                                            hist_title=hist_title,
                                            spacing_title=spacing_title,
                                            hist_dir=self.egg_hist_dir,
                                            spacing_dir=self.egg_spacing_dir)

    def _load_anno_data(self, obj):
        data = pd.read_csv(self.anno_data_path, header=0)
        data = data[data['New'] == 'yes']
        try:
            data = data[data['Object'].str.contains(obj)]
            data.sort_values(by=['LaneNum', 'VidNum', 'VidTime'],
                             inplace=True)

            print(f'{obj.capitalize()} data loaded')
            self.no_blanks_flag = True
        except ValueError:
            self.no_blanks_flag = False
            print(f'\nError loading data. A new object entry in the '
                  '\nannotations file has a blank label. Please delete '
                  '\nthat entry or fill it in with the correct object label.')

        return data

    def _load_t_data(self):
        try:
            data = pd.read_csv(self.t_data_path, header=0)
            print('Temperature data loaded')
            self.t_data_flag = True
        except Exception:
            self.t_data_flag = False
            print('\nUnknown error loading temperature data')

        return data

    def _make_dirs(self):
        os.mkdir(self.ana_dir)
        print('\nAnalysis directory created')
        os.mkdir(self.egg_graphs_dir)
        os.mkdir(self.egg_hist_dir)
        os.mkdir(self.egg_spacing_dir)
        print('Egg graph directories created')
        os.mkdir(self.t_dir)
        print('Temperature directory created\n')

    def _process_egg_lanes(self):
        self.min_obj_num = int(self.minEggsLineEdit.text())

        self.egg_lane_dfs = []

        try:
            max_lane_num = int(self.egg_data['LaneNum'].max())
        except ValueError:
            print(f'This experiment may not have any eggs.'
                  'Check annotations file.')

        # Get and clean egg interval data for each lane
        for i in range(1, 32+1):
            lane_df = self.egg_data[self.egg_data['LaneNum'] == i].copy()

            # Need to initially reset index to deal with
            # "ValueError: cannot reindex from a duplicate axis"
            lane_df = lane_df.reset_index(drop=True)

            # Separate any egg cluster into single egg entries
            for j, row in enumerate(lane_df.itertuples()):
                object = row.Object
                if 'eggs' in object:
                    num_eggs = int(object.split()[0])

                    lane_df.loc[j, 'Object'] = 'egg'
                    row_copy = lane_df.iloc[j]

                    for e in range(1, num_eggs):
                        added_time = e * 0.0001
                        lane_df = lane_df.append(row_copy)
                        lane_df.iloc[-1,
                                     lane_df.columns.get_loc(
                                      'SecFromExperimentStart')] = \
                            row.SecFromExperimentStart + added_time

            lane_df.sort_values(by=['LaneNum', 'VidNum', 'VidTime'],
                                inplace=True)

            lane_df = lane_df.reset_index(drop=True)

            # Calculate intervals
            lane_df['intervals'] = lane_df['SecFromExperimentStart'].diff()

            # Deal with 0 or negative values in intervals
            zeros = lane_df[lane_df['intervals'] == 0]
            if not zeros.empty:
                print('Zeros in lane', i, 'intervals')
                for k, row in zeros.iterrows():
                    lane_df.loc[k, 'intervals'] = 0.0001

            negs = lane_df[lane_df['intervals'] < 0]
            if not negs.empty:
                print('Negatives in lane', i, 'intervals')
                print('Sorting is not working correctly')

            # Calculate log intervals
            lane_df['log_intervals'] = \
                   [np.log(x) for x in lane_df['intervals']]

            # Throw out lanes with fewer than min_obj_num entries
            if len(lane_df) < self.min_obj_num:
                # Clear dataframe contents, keep header
                lane_df = lane_df.iloc[0:0]

            self.egg_lane_dfs.append(lane_df)

        self._plot_eggs(self.egg_lane_dfs)

    def _process_l1_lanes(self):
        self.l1_lane_dfs = []

        try:
            max_lane_num = int(self.l1_data['LaneNum'].max())
        except ValueError:
            print(f'This experiment may not have any l1s. '
                  'Check annotations file.')

        # Get l1 info for each lane
        for i in range(1, 32+1):
            lane_df = self.l1_data[self.l1_data['LaneNum'] == i].copy()

            # Need to initially reset index to deal with
            # "ValueError: cannot reindex from a duplicate axis"
            lane_df = lane_df.reset_index(drop=True)

            lane_df.sort_values(by=['LaneNum', 'VidNum', 'VidTime'],
                                inplace=True)

            lane_df = lane_df.reset_index(drop=True)

            self.l1_lane_dfs.append(lane_df)

    def _populate_out_files(self):
        lanes_list = list(range(32))
        lanes_dic = dict.fromkeys(lanes_list, None)

        info_file_header = ['p',
                            'lambda-1',
                            'lambda-2',
                            'EggCount',
                            'SFES',
                            'Intervals',
                            'FirstL1Hour',
                            'L1Count']
        info_file_str = (f'{self.ana_dir}/{self.exp_name}'
                         '_analysis_results.csv')

        # Create data file
        funcs.access_csv(info_file_str, [info_file_header], 'w')

        # Get egg param estimates, counts, and SFES
        for i, df in enumerate(self.egg_lane_dfs):
            if df.empty:
                lanes_dic[i] = ['NA', 'NA', 'NA', 0, 'NA', 'NA']
            else:
                params = funcs.estimate_parameters(df['intervals'])
                params.append(len(df))  # obj count = len(df)
                sfes = list(df.SecFromExperimentStart)
                params.append([round(x, 4) for x in sfes])
                params.append([round(x, 4) for x in df.loc[1:, 'intervals']])
                lanes_dic[i] = params

        # Get l1 timings and counts
        for i, df in enumerate(self.l1_lane_dfs):
            if df.empty:
                lanes_dic[i].extend(['NA', 0])
            else:
                first_row = df.iloc[0]
                first_l1_sec = first_row.SecFromExperimentStart
                first_l1_min = round(first_l1_sec * (1 / 60), 2)
                first_l1_hr = round(first_l1_min * (1 / 60), 2)
                l1_info = [first_l1_hr]
                l1_info.append(len(df))  # count = len(df)
                lanes_dic[i].extend(l1_info)

        # Save data
        for key in lanes_dic:
            funcs.access_csv(info_file_str, [lanes_dic[key]], 'a')

    def _process_temperature(self):
        t_data_header = ['Min', 'Max', 'Mean', 'STD']

        # Create temperature file and set header
        funcs.access_csv(self.t_summary_csv_str,
                         [t_data_header],
                         'w')

        temp_stats = funcs.get_temp_stats(self.t_data)

        funcs.access_csv(self.t_summary_csv_str,
                         [temp_stats],
                         'a')

        funcs.plot_temperatures(self.t_data,
                                save=True,
                                save_dir=self.t_dir,
                                title_str=f'{self.exp_name} Temperatures')

    def main(self, exp_path):
        try:
            self._set_paths(exp_path)
            if REANALYSIS_FLAG:
                shutil.rmtree(self.ana_dir)
            self._make_dirs()
            # Load main data early to check for blank object entries
            # and to have it ready for processing
            self.egg_data = self._load_anno_data('egg')
            self.l1_data = self._load_anno_data('l1')
            self.t_data = self._load_t_data()
            self.no_prev_ana_flag = True
        except FileExistsError:
            self.no_prev_ana_flag = False
            print('\nAnalysis directory already exists. Please delete the'
                  '\nanalysis directory and click "Analyze" if you want to'
                  '\nanalyze the experiment again.')

        if self.no_prev_ana_flag:
            if self.no_blanks_flag:
                if self.t_data_flag:
                    print('\nProcessing...\n')
                    self._process_egg_lanes()
                    self._process_l1_lanes()
                    self._populate_out_files()
                    self._process_temperature()
                    print('\nAnalysis complete')

    def analyze(self):
        for t_dir in os.listdir(ACT_EXP_DIR):
            dir_path = f'{ACT_EXP_DIR}{t_dir}'
            if os.path.isdir(dir_path):
                for exp_dir in os.listdir(dir_path):
                    exp_path = f'{dir_path}/{exp_dir}'
                    self.main(exp_path)
        print('\n\nDone')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = EggAnaGUI()
    main_gui.show()
    sys.exit(app.exec_())
