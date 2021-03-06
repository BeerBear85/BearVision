# File: user.py
# A class for holder the info releated to a specific user of the system

import os, datetime, csv, logging
import numpy as np
import pandas as pd
import GPS_Functions
import BasicClipSpecification
from InputGPS_Importer import InputGPS_Importer
from ConfigurationHandler import ConfigurationHandler
from Enums import ClipTypes

logger = logging.getLogger(__name__)

class User:
    def __init__(self, arg_user_folder):
        logger.info("Creating user: " + arg_user_folder.name)
        tmp_options = ConfigurationHandler.get_configuration()
        self.name = arg_user_folder.name
        self.user_folder = arg_user_folder.path
        self.user_GPS_input_files = os.path.join(self.user_folder, tmp_options['USER']['user_GPS_input_files_subpath'])
        self.location_data_folder = os.path.join(self.user_folder, tmp_options['USER']['location_data_subpath'])
        self.full_clip_output_video_folder = os.path.join(self.user_folder, tmp_options['USER']['full_clip_output_video_subpath'])
        self.tracker_clip_output_video_folder = os.path.join(self.user_folder, tmp_options['USER']['tracker_clip_output_video_subpath'])
        self.maximum_distance = float(tmp_options['USER']['maximum_distance'])
        self.minimum_velocity = float(tmp_options['USER']['minimum_velocity']) / 3.6  # convert from km/h to m/s
        self.time_search_range = datetime.timedelta(seconds=float(tmp_options['USER']['time_search_range']))

        self.location_data_names = ['time', 'latitude', 'longitude', 'velocity', 'precision', 'satellites']
        self.location_data = pd.DataFrame(columns=self.location_data_names)

        self.obstacle_match_data_names = ['time', 'video_file', 'init_bbox']
        self.obstacle_match_data = pd.DataFrame(columns=self.obstacle_match_data_names)

        self.refresh_gps_data()

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

        self.location_data = self.location_data[~self.location_data.index.duplicated(keep='first')]  # Remove multiple entries with the same time stamp be bnexe3
        #print('Full location data:' + str(self.location_data))
        return

    def is_close(self, target_date, target_location):
        #logger.debug("Types: " + str(type(target_date)) + "  " + str(type(target_location)))
        #print("Before: " + (target_date - self.time_search_range).strftime("%Y%m%d_%H_%M_%S_%f"))
        #print("After:  " + (target_date + self.time_search_range).strftime("%Y%m%d_%H_%M_%S_%f"))
        log_data_within_time_interval = self.location_data.truncate(before=target_date - self.time_search_range, after=target_date + self.time_search_range)

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

    def add_obstacle_match(self, arg_start_time_entry, arg_video_file, arg_init_bbox):
        data_entry = [arg_start_time_entry, arg_video_file, arg_init_bbox] # format for panda frame (table)
        data_entry = pd.DataFrame([data_entry], columns=self.obstacle_match_data_names)
        #self.obstacle_match_data = self.obstacle_match_data.append(data_entry)
        self.obstacle_match_data = pd.concat([self.obstacle_match_data, data_entry])   # concat all the match data
        return

    def filter_obstacle_matches(self, arg_user_match_minimum_interval : float):
        # Drop every match which is closer (after) a match than the minimum interval
        self.obstacle_match_data.sort_values(by='time', inplace=True)
        self.obstacle_match_data.index = range(len(self.obstacle_match_data))  # reset indexing of dataframe
        tmp_match_minimim_interval = datetime.timedelta(seconds=int(arg_user_match_minimum_interval))
        tmp_indexes_to_drop = []

        #logger.debug("Pre-filtered match list of user %s, is: %s", self.name, str(self.obstacle_match_data))

        # This is probably not the smartes way of doing this!
        for index, row in self.obstacle_match_data.iterrows():
            detection_time = row["time"]

            if ((index+1) < self.obstacle_match_data.shape[0]):  # not out of range
                if self.obstacle_match_data.at[index+1,'time'] < (detection_time + tmp_match_minimim_interval): #As the dataframe is sorted, we already know that the next item is later in time
                    tmp_indexes_to_drop.append(index+1)

        self.obstacle_match_data.drop(tmp_indexes_to_drop, inplace=True)

        logger.debug("Filtered match list for user %s, is: %s", self.name,str(self.obstacle_match_data))
        return

    # Creates a list of BasicClipSpecification objects for known matches of the user
    def create_clip_specifications(self, clip_type : ClipTypes):
        clip_spec_list = []
        video_output_path = ''
        for index, obstacle_row in self.obstacle_match_data.iterrows():
            time_str = obstacle_row["time"].strftime("%Y%m%d_%H_%M_%S")
            video_output_name_short = self.name + "_" + time_str + ".avi"

            if clip_type is ClipTypes.FULL_CLIP:
                video_output_path = os.path.join(self.full_clip_output_video_folder, video_output_name_short)
            elif clip_type is ClipTypes.TRACKER_CLIP:
                video_output_path = os.path.join(self.tracker_clip_output_video_folder, video_output_name_short)

            if not os.path.exists(video_output_path):
                clip_spec = BasicClipSpecification.BasicClipSpecification(obstacle_row["video_file"].path,
                                                                          obstacle_row["time"],
                                                                          video_output_path,
                                                                          obstacle_row["init_bbox"])
                clip_spec_list.append(clip_spec)
                logger.debug("Entry in list of new clip_specification_list" + clip_spec.output_video_path)

        return clip_spec_list

    def __nearest_date(self, date_list, target_date):  # Quite slow, so should only be used on a limited amount of data
        nearest = min(date_list, key=lambda x: abs(x - target_date))
        timedelta = abs(nearest - target_date)
        return nearest, timedelta
