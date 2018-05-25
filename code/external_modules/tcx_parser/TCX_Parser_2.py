import logging, os, csv, re
import xml.etree.ElementTree
import pandas as pd

logger = logging.getLogger(__name__)  #Set logger to reflect the current file

# Note: horizontal_dilution and number of satellites are not present in this file type.

class TCXParser2:
    def __init__(self, arg_input_file: os.DirEntry):
        tree = xml.etree.ElementTree.parse(arg_input_file.path)
        root = tree.getroot()

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
                        time = trackpoint.find(root_ns + 'Time').text.strip()
                    except:
                        time = ''
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



        return
