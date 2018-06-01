# todo:
# beregn afstand
# find om afstand er inden for krav
# lavpass filrer velocity
# tilføj krav om min hastighed


import numpy as np
import matplotlib.pyplot as plt
import datetime, os, re, time, sys

sys.path.append('..\modules')
import User  # AppUser
import GPS_Functions

# Consider using mmodule "configparser"
# inbox_video_directory = '../clip_extractor/outputs/sep_18_clips'
# clock_correction = datetime.timedelta(seconds = -3) #How much the camera is off (use GPS timestamp instead)
inbox_video_directory = os.path.abspath('../preprocessor/outputs')
clock_correction = datetime.timedelta(seconds=10)  # GPS

maximum_distance = 30  # [m]

# camera_location = [55.682506, 12.623083]
jump_approach_location = [55.682366, 12.623255]  # lat/long

# Create users and read GPS data files
test_user = User.AppUser('bear')
test_user.refresh_gps_data()

# Debug plot
ax = plt.gca()
ax.cla()

# Read inbox
for inbox_entry in os.scandir(inbox_video_directory):
    if inbox_entry.name.endswith('.MP4') and inbox_entry.is_file():
        # print("Checking for GPS matches on file: " + inbox_entry.name)
        stopwatch_start = time.time()
        inbox_entry_start_time = datetime.datetime.strptime(re.findall("\d+\_\d+\_\d+\_\d+", inbox_entry.name)[0], "%Y%m%d_%H_%M_%S")

        #print("Video time: " + inbox_entry_start_time.strftime("%Y%m%d_%H_%M_%S"))
        if test_user.is_close(inbox_entry_start_time + clock_correction, jump_approach_location):
            print('Clip name: ' + inbox_entry.name)
            test_user.copy_full_clip(os.path.join(inbox_video_directory, inbox_entry.name))

        # duration = (time.time()-float(stopwatch_start))
        # print('Elapsed time:' + str(duration) + ' seconds')

        # if ((nearest_delta <= pa_maximum_time_tolerance) and (video_dist <= pa_maximum_distance) and (video_vel >= pa_minimum_velocity)):
        #	print('File:' + inbox_entry.name + 'Time:' + str(inbox_entry_start_time) + ' Nearest:' + nearest_str + ' Dist:' + str(video_dist) + ' Vel:' + str(video_vel))
        #	[x_dist, y_dist] = np.array(GPS_Functions.get_relative_coordinates(jump_approach_location[0], jump_approach_location[1], video_lat, video_long))

        #	ax.plot(x_dist, y_dist,'kd', markersize=14)
        #	ax.text(x_dist, y_dist, nearest_str)

print("Finished looking through files")

### Print full track ###
lat_meas = test_user.data.as_matrix(columns=['latitude'])
long_meas = test_user.data.as_matrix(columns=['longitude'])
[x_dist, y_dist] = np.array(
    GPS_Functions.get_relative_coordinates(jump_approach_location[0], jump_approach_location[1], lat_meas, long_meas))

ax.plot(x_dist, y_dist, 'b-', 0, 0, 'rx')
circle_ref = plt.Circle((0, 0), maximum_distance)
ax.add_artist(circle_ref)
plt.show()

# Calculate distance to jump
