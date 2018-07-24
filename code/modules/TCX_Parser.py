import logging, os, csv, re
import datetime as dt
import xml.etree.ElementTree
import pandas as pd
from ConfigurationHandler import ConfigurationHandler

logger = logging.getLogger(__name__)  # Set logger to reflect the current file

# Note: horizontal_dilution and number of satellites are not present in this file type.


class TCXParser:
    def __init__(self, arg_input_file: os.DirEntry):
        tmp_options = ConfigurationHandler.get_configuration()
        tree = xml.etree.ElementTree.parse(arg_input_file.path)
        root = tree.getroot()
        data_names = ['time', 'latitude', 'longitude', 'speed', 'horizontal_dilution', 'satellites']
        self.data = pd.DataFrame()

        # Namespaces of the XML fields
        root_ns = "{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
        ns3 = "{http://www.garmin.com/xmlschemas/ActivityExtension/v2}"

        if root.tag != root_ns + 'TrainingCenterDatabase':
            print('Unknown root found: ' + root.tag)
            return
        activities = root.find(root_ns + 'Activities')
        if not activities:
            print('Unable to find Activities under root')
            return
        activity = activities.find(root_ns + 'Activity')
        if not activity:
            print('Unable to find Activity under Activities')
            return
        for lap in activity.iter(root_ns + 'Lap'):
            for track in lap.iter(root_ns + 'Track'):
                for trackpoint in track.iter(root_ns + 'Trackpoint'):
                    try:
                        time_str = trackpoint.find(root_ns + 'Time').text.strip()
                    except:
                        time_str = '2000-01-01T00:00:00.000Z'
                    # Format example: 2018-05-15T15:43:44.000Z
                    time = dt.datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                    try:
                        latitude = trackpoint.find(root_ns + 'Position').find(root_ns + 'LatitudeDegrees').text.strip()
                    except:
                        latitude = ''
                    try:
                        longitude = trackpoint.find(root_ns + 'Position').find(root_ns + 'LongitudeDegrees').text.strip()
                    except:
                        longitude = ''
                    try:
                        speed = trackpoint.find(root_ns + 'Extensions').find(ns3 + 'TPX')\
                                .find(ns3 + 'Speed').text.strip()
                    except:
                        speed = ''
                    #  logger.debug("Speed entry: " + speed)

                    #Add the extracted data to the panda dataframe
                    dummy_hdop = float(tmp_options['GPS_FILE_PARSING']['TCX_dummy_hdop']) # [m]
                    dummy_satellites = int(tmp_options['GPS_FILE_PARSING']['TCX_dummy_satellites'])  # [-]
                    data_entry = [time.strftime("%Y%m%d_%H_%M_%S_%f"), latitude, longitude, speed, dummy_hdop, dummy_satellites]  # format for panda frame (table)
                    data_entry = pd.DataFrame([data_entry], columns=data_names)
                    self.data = pd.concat([self.data, data_entry])  # I'm sure this is not the smartest way to do this

