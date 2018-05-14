# File: user.py
# A class for holder the info releated to a specific user of the system

import os, datetime, csv, shutil
import numpy as np
import pandas as pd
import GPS_Functions

# Program start initilisation
user_data_path      = '../users'
gps_data_subpath    = 'gps_data'
full_clips_subpath  = 'full_clips'

maximum_distance = 30  # [m]
minimum_velocity = 10 / 3.6  # [m/s]
maximum_time_tolerance = datetime.timedelta(seconds=10)  # Only accept GPS measurement within this time frame


class AppUser:
    def __init__(self, arg_name):

        print("Creating user: " + arg_name)
        self.name = arg_name
        self.user_folder = os.path.join(user_data_path, arg_name)
        self.gps_data_folder = os.path.join(self.user_folder, gps_data_subpath)

        self.data_names = ['time', 'latitude', 'longitude', 'velocity', 'precision', 'satellites']
        self.data = pd.DataFrame()

    # print(self.data)

    def refresh_gps_data(self):
        self.data = pd.DataFrame()  # reset
        for gps_file in os.scandir(self.gps_data_folder):
            if gps_file.name.endswith('.csv') and gps_file.is_file():
                gps_file_full = os.path.join(self.gps_data_folder, gps_file.name)
                #print("GPS file: " + gps_file_full)
                file_data = pd.read_csv(gps_file_full, index_col='time', names=self.data_names)
                file_data.index = pd.to_datetime(file_data.index, format="%Y%m%d_%H_%M_%S_%f")
                # print(file_data)
                self.data = pd.concat([self.data, file_data])

        #print('Full data:' + str(self.data))
        return

    def is_close(self, target_date, target_location):
        log_data_within_time_interval = self.data.truncate(before=target_date - maximum_time_tolerance, after=target_date + maximum_time_tolerance)
        #print("Before: " + (target_date - maximum_time_tolerance).strftime("%Y%m%d_%H_%M_%S_%f"))
        #print("After:  " + (target_date + maximum_time_tolerance).strftime("%Y%m%d_%H_%M_%S_%f"))


        if not log_data_within_time_interval.empty:  # not empty
            [nearest, nearest_time_delta] = self.nearest_date(log_data_within_time_interval.index, target_date)
            #print("Nearest: " + str(nearest))
            nearest_location = [log_data_within_time_interval.loc[nearest, 'latitude'],
                                log_data_within_time_interval.loc[nearest, 'longitude']]
            nearest_velocity = log_data_within_time_interval.loc[nearest, 'velocity']
            nearest_distance = GPS_Functions.get_distance(target_location[0], target_location[1], nearest_location[0],nearest_location[1])
            #print("Nearest dist/vel: " + str(nearest_distance) + " " + str(nearest_velocity))
            if ((nearest_distance <= maximum_distance) and (nearest_velocity >= minimum_velocity)):
                nearest_time_str = nearest.strftime("%Y%m%d_%H_%M_%S")
                print('Found file at nearest time: ' + nearest_time_str + ' Dist: ' + str(
                    nearest_distance) + ' Vel: ' + str(nearest_velocity))
                return True  # maybe also get dataframe row of the 'nearest' element
        return False

    def nearest_date(self, date_list, target_date):  # Quite slow, so should only be used on a limited amount of data
        nearest = min(date_list, key=lambda x: abs(x - target_date))
        timedelta = abs(nearest - target_date)
        return nearest, timedelta

    def copy_full_clip(self, arg_source_full_path):
        base_filename = os.path.basename(arg_source_full_path)
        destination = os.path.join(self.user_folder, full_clips_subpath, base_filename)
        shutil.copy(arg_source_full_path, destination)
