# File: user.py
# A class for holder the info releated to a specific user of the system

import os, datetime, csv, logging
import numpy as np
import pandas as pd
import GPS_Functions
import FullClipSpecification
from InputGPS_Importer import InputGPS_Importer
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)

class User:
    def __init__(self, arg_user_folder):
        logger.info("Creating user: " + arg_user_folder.name)
        tmp_options = ConfigurationHandler.get_configuration()
        self.name = arg_user_folder.name
        self.user_folder = arg_user_folder.path
        self.user_GPS_input_files = os.path.join(self.user_folder, tmp_options['USER']['user_GPS_input_files_subpath'])
        self.location_data_folder = os.path.join(self.user_folder, tmp_options['USER']['location_data_subpath'])
        self.output_video_folder = os.path.join(self.user_folder, tmp_options['USER']['output_video_subpath'])
        self.maximum_distance = float(tmp_options['USER']['maximum_distance'])
        self.minimum_velocity = float(tmp_options['USER']['minimum_velocity']) / 3.6  # convert from km/h to m/s
        self.time_search_range = datetime.timedelta(seconds=float(tmp_options['USER']['time_search_range']))

        self.location_data_names = ['time', 'latitude', 'longitude', 'velocity', 'precision', 'satellites']
        self.location_data = pd.DataFrame()

        self.obstacle_match_data_names = ['time', 'video_file']
        self.obstacle_match_data = pd.DataFrame()

    def refresh_gps_data(self):
        InputGPS_Importer.import_user_format_gps_files(self.user_GPS_input_files, self.location_data_folder)  # creates BearVison formatted GPS files
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

        self.location_data = self.location_data[~self.location_data.index.duplicated(keep='first')]  # Remove multiple entries with the same time stamp
        #print('Full data:' + str(self.data))
        return

    def is_close(self, target_date, target_location):
        #logger.debug("Types: " + str(type(target_date)) + "  " + str(type(target_location)))
        log_data_within_time_interval = self.location_data.truncate(before=target_date - self.time_search_range, after=target_date + self.time_search_range)
        #print("Before: " + (target_date - self.time_search_range).strftime("%Y%m%d_%H_%M_%S_%f"))
        #print("After:  " + (target_date + self.time_search_range).strftime("%Y%m%d_%H_%M_%S_%f"))

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

            if (nearest_distance <= self.maximum_distance) and (nearest_velocity >= self.minimum_velocity):
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
            video_output_name_short = self.name + "_" + time_str + ".avi"
            video_output_path = os.path.join(self.output_video_folder, video_output_name_short)
            if not os.path.exists(video_output_path):
                full_clip_spec = FullClipSpecification.FullClipSpecification(row["video_file"], row["time"], video_output_path)
                full_clip_spec_list.append(full_clip_spec)
                logger.debug("Entry in list of new clip_specification_list" + full_clip_spec.output_video_path)
        return full_clip_spec_list

    def __nearest_date(self, date_list, target_date):  # Quite slow, so should only be used on a limited amount of data
        nearest = min(date_list, key=lambda x: abs(x - target_date))
        timedelta = abs(nearest - target_date)
        return nearest, timedelta
