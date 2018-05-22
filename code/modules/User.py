# File: user.py
# A class for holder the info releated to a specific user of the system

import os, datetime, csv, shutil, logging
import numpy as np
import pandas as pd
import GPS_Functions
import FullClipSpecification

logger = logging.getLogger(__name__)

# Program start initilisation
location_data_subpath = 'internal_files/location_info'
output_video_subpath = 'output_video_files'

maximum_distance = 30  # [m]
minimum_velocity = 10 / 3.6  # [m/s]
maximum_time_tolerance = datetime.timedelta(seconds=10)  # Only accept GPS measurement within this time frame


class User:
    def __init__(self, arg_user_folder):
        logger.info("Creating user: " + arg_user_folder.name)
        self.name = arg_user_folder.name
        self.user_folder = arg_user_folder.path
        self.location_data_folder = os.path.join(self.user_folder, location_data_subpath)
        self.output_video_folder = os.path.join(self.user_folder, output_video_subpath)

        self.location_data_names = ['time', 'latitude', 'longitude', 'velocity', 'precision', 'satellites']
        self.location_data = pd.DataFrame()

        self.obstacle_match_data_names = ['time', 'video_file']
        self.obstacle_match_data = pd.DataFrame()

        self.refresh_gps_data()

    def refresh_gps_data(self):
        self.location_data = pd.DataFrame()  # reset
        for gps_file in os.scandir(self.location_data_folder):
            if gps_file.name.endswith('.csv') and gps_file.is_file():
                gps_file_full = os.path.join(self.location_data_folder, gps_file.name)
                logger.debug("Reading GPS file: " + gps_file_full)
                file_data = pd.read_csv(gps_file_full, index_col='time', names=self.location_data_names)  # Use time for indexing
                # If the GPS file does not have this high precision there is a chance to get dublicated entries
                file_data.index = pd.to_datetime(file_data.index, format="%Y%m%d_%H_%M_%S_%f")
                # print(file_data)
                self.location_data = pd.concat([self.location_data, file_data]) # concat all the read GPS data

        #print('Full data:' + str(self.data))
        return

    def is_close(self, target_date, target_location):
        #logger.debug("Types: " + str(type(target_date)) + "  " + str(type(target_location)))
        log_data_within_time_interval = self.location_data.truncate(before=target_date - maximum_time_tolerance, after=target_date + maximum_time_tolerance)
        #print("Before: " + (target_date - maximum_time_tolerance).strftime("%Y%m%d_%H_%M_%S_%f"))
        #print("After:  " + (target_date + maximum_time_tolerance).strftime("%Y%m%d_%H_%M_%S_%f"))

        if not log_data_within_time_interval.empty:  # not empty
            [nearest, nearest_time_delta] = self.__nearest_date(log_data_within_time_interval.index, target_date)
            #logger.debug("nearest info: " + str(type(nearest)) + "  " + str(nearest))
            # Remember that the time is used for indexing the panda "table"
            nearest_location = [log_data_within_time_interval.loc[nearest, 'latitude'],
                                log_data_within_time_interval.loc[nearest, 'longitude']]
            nearest_velocity = log_data_within_time_interval.loc[nearest, 'velocity']
            nearest_distance = GPS_Functions.get_distance(target_location[0], target_location[1], nearest_location[0], nearest_location[1])
            #logger.debug("Nearest dist/vel: " + str(nearest_distance) + " " + str(nearest_velocity))
            #logger.debug("nearest_velocity info: " + str(type(nearest_velocity)) + "  " + str(nearest_velocity))

            #  Dublicated entries should be handled better
            if (nearest_distance <= maximum_distance) and (nearest_velocity >= minimum_velocity):
                nearest_time_str = nearest.strftime("%Y%m%d_%H_%M_%S")
                logger.debug('Found file at nearest time: ' + nearest_time_str + ' Dist: ' + str(
                    nearest_distance) + ' Vel: ' + str(nearest_velocity))
                return True  # maybe also get dataframe row of the 'nearest' element
        return False

    def add_obstacle_match(self, arg_start_time_entry, arg_video_file):
        data_entry = [arg_start_time_entry, arg_video_file] # format for panda frame (table)
        data_entry = pd.DataFrame([data_entry], columns=self.obstacle_match_data_names)
        self.obstacle_match_data = pd.concat([self.obstacle_match_data, data_entry])   # concat all the match data
        return

    # Creates a list of FullClipSpecification objects for known matches of the user
    def create_full_clip_specifications(self):
        full_clip_spec_list = []
        for index, row in self.obstacle_match_data.iterrows():
            time_str = row["time"].strftime("%Y%m%d_%H_%M_%S")
            output_name = self.name + "_" + time_str
            full_clip_spec = FullClipSpecification.FullClipSpecification(row["video_file"], row["time"], output_name)
            full_clip_spec_list.append(full_clip_spec)
        return full_clip_spec_list

    def __nearest_date(self, date_list, target_date):  # Quite slow, so should only be used on a limited amount of data
        nearest = min(date_list, key=lambda x: abs(x - target_date))
        timedelta = abs(nearest - target_date)
        return nearest, timedelta

    def copy_full_clip(self, arg_source_full_path):
        base_filename = os.path.basename(arg_source_full_path)
        destination = os.path.join(self.output_video_folder, base_filename)
        shutil.copy(arg_source_full_path, destination)

