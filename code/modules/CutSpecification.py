# File: CutSpecification.py
# A class for parsing the "bboxes" file

import fileinput
import re
import numpy as np

class CutSpecification:
    def __init__(self, arg_spec_filename):
        self.spec_filename = arg_spec_filename
      
      
         self.video_filename = ''
         self.track_id = 0
         self.start_frame = 0
         self.track_age = 0
         self.box_dimensions = [0, 0]
         self.box_coordinates = [0, 0]
      
         self.parse_file()
      
      
    def parse_file(self):
   
        #Consider using mmodule "configparser"
        textfile = open(self.spec_filename, 'r') #maybe require "rb"
        filetext = textfile.read()
        textfile.close()
      
        self.video_filename = re.findall("(?<=Video filename: )[\S\ ]+", filetext)[0]
        self.track_id       = int(re.findall("(?<=Track id: )\S+", filetext)[0])
        self.start_frame    = int(re.findall("(?<=Start frame: )\S+", filetext)[0])
        self.track_age      = int(re.findall("(?<=Track age: )\S+", filetext)[0])
        self.box_dimensions = np.fromstring(re.findall("(?<=Bounding box dimension: )\S+", filetext)[0], dtype=int, sep=',')
      
        #Read coordinate vector in the bottom of the file
        textfile = open(self.spec_filename, 'rb')
        self.box_coordinates = np.loadtxt(textfile, dtype=int, delimiter=",", skiprows=6)
      
#      print(self.track_id)
#      print(self.box_dimensions)
#      print(self.box_coordinates)
         
      
