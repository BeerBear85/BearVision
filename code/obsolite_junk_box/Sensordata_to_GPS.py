#Converts from the Sensordata format to normal GPS CSV format

import argparse, csv, os, re, datetime

verbose = True

#### Handle commandline arguments ####
parser = argparse.ArgumentParser(description='Convert from SensorData app format to GPS csv format')
parser.add_argument('--path', required=True, help='Path to SensorData file folder')
args = parser.parse_args()

time_range_filename = os.path.join(args.path, '_info.json')

if (verbose):
	print("Opening file: " + time_range_filename)


textfile = open(time_range_filename, 'r') #maybe require "rb"
filetext = textfile.read()
textfile.close()

if (verbose):
	print("filetext: " + filetext)

start_time_ms  = int(re.findall(r"(?<=\"start\":)\d+", filetext)[0])

if (verbose):
	print("start_time_ms: " + str(start_time_ms))

epoch_data = datetime.datetime(1970, 1, 1)
time_zone_offset_hour = 0
start_time = epoch_data + datetime.timedelta(hours = time_zone_offset_hour, milliseconds = start_time_ms) #datetime
start_time_str = start_time.strftime("%Y%m%d_%H_%M_%S")
if (verbose):
	print("start_time string: " + start_time_str)

gps_file = os.path.join(args.path, 'gps.csv')
with open(gps_file, 'r') as csvfile:
	reader = csv.reader(csvfile)
	input_list = list(reader)

if (verbose):
	print("The first few lines of the GPS file:")
	print(input_list[0])
	print(input_list[1])
	print(input_list[2])

output_file = os.path.join(args.path, (start_time_str + '_vision_bear_GPS.csv'))
with open(output_file, 'w', newline='') as csvfile:
	output_writer = csv.writer(csvfile)

	time       = list()
	latitude   = list()
	longitude  = list()
	accuracy   = list()
	speed      = list()
	satellites = list()
	for i in range(1,len(input_list)):
		time      .append( start_time + datetime.timedelta(milliseconds = int(input_list[i][0])) )
		latitude  .append( float(input_list[i][1]) )
		longitude .append( float(input_list[i][2]) )
		accuracy  .append( float(input_list[i][3]) )
		speed     .append( float(input_list[i][5]) )
		satellites.append(   int(input_list[i][7]) )
		
		output_writer.writerow( [time[i-1].strftime("%Y%m%d_%H_%M_%S_%f"), str(latitude[i-1]), str(longitude[i-1]), str(speed[i-1]), str(accuracy[i-1]), str(satellites[i-1])] )

#		if (input_list[i][0] == "3291575") or (input_list[i][0] == "3292573"):
#			print("start time: " + str(start_time))
#			print("milliseconds: " + str(int(input_list[i][0])))
#			print("Time string: " + time[i-1].strftime("%Y%m%d_%H_%M_%S_%f"))

		#if (verbose):
		#	print(time[i-1].strftime("%Y%m%d_%H_%M_%S"), str(latitude[i-1]), str(longitude[i-1]), str(speed[i-1]))
		
