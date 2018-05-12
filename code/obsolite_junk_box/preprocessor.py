
import argparse, cv2, time, os, datetime, sys
import numpy as np
sys.path.append('..\modules')
import GoproVideo

#Program parameters
inbox_video_directory = '../video/2017/sep_27'#/GP020491.MP4'
output_path = '.\outputs'

processing_fps = 15
start_frame = int(100)
morph_open_size = 20
allowed_clip_interval = 5 #[s]
start_caption_offset = 0.5 #[s] rewind offset for when capturing clip
clip_duration = 6 #[s]
motion_frame_counter_threshold = 3

#Initilise
next_allowed_frame = 0
motion_frame_counter = 0
input_video = GoproVideo.GoproVideo()
foreground_extractor_GMG = cv2.bgsegm.createBackgroundSubtractorGMG( initializationFrames = 60, decisionThreshold = 0.8 )
morph_open_kernel = np.ones((morph_open_size,morph_open_size),np.uint8)

total_stopwatch_start = time.time()
for inbox_entry in os.scandir(inbox_video_directory):
	
	if inbox_entry.name.endswith('.MP4') and inbox_entry.is_file():
		video_stopwatch_start = time.time()
		stopwatch_start = time.time()
		
		input_video_name = os.path.join(inbox_video_directory, inbox_entry.name)
		print('Preprocissiong video: ', input_video_name)
		
		### Initilise video info ###
		#input_video = GoproVideo.GoproVideo() #debug
		next_allowed_frame = 0
		input_video.init(input_video_name)
		
		
		processing_frame_interval = int((input_video.fps+1)/processing_fps)
		print('output inteval: ' + str(input_video.fps))
		
		print('Initilisation time:' + str((time.time()-float(stopwatch_start))) + ' seconds')
		
		#print('Num of frames: ' + str(int(input_video.frames)) )# + 'float: ' str(input_video.frames))
		### Read frames and write cut_out ###
		input_video.set_start_point(start_frame) #Set frame point of video
		for frame_number in range(start_frame, int(input_video.frames)):#, processing_frame_interval):
			#print('Processing frame number: ' + str(frame_number))
			#stopwatch_start = time.time()
			#input_video.set_start_point(frame_number) #Set frame point of video
			#print('Startpoint setting time:' + str((time.time()-float(stopwatch_start))) + ' seconds')
			
			frame_stopwatch_start = time.time()
			read_return_value, frame = input_video.read_frame()
			if (read_return_value == 0): #end of file
				print('End of file!!!')
				break

			if (frame_number%processing_frame_interval == 0): #For processing FPS
				frame_cut = frame[650:950, 1:300]
				#Resize the frame
				#frame_cut = cv2.resize(frame_cut, None, fx=0.50, fy=0.50, interpolation = cv2.INTER_LINEAR )
				mask = foreground_extractor_GMG.apply(frame_cut)
				
				#mask_MOG2 = foreground_extractor_MOG2.apply(frame_cut)
				#mask_GMG  = foreground_extractor_GMG.apply(frame_cut)
				#cv2.imshow('Frame Cut',frame_cut)

				#cv2.imshow('Mask MOG 2',mask_MOG2)
				#cv2.imshow('Mask', mask)
				mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_open_kernel)
				#cv2.imshow('Mask morphed', mask)
				#cv2.waitKey(10)

				if (mask.any() and frame_number > next_allowed_frame):
					motion_frame_counter = motion_frame_counter + 1
					if (motion_frame_counter > motion_frame_counter_threshold):
						print("Found something moving at frame " + str(frame_number))
						motion_frame_counter = 0
						next_allowed_frame = frame_number + int(input_video.fps*allowed_clip_interval)
						relative_start_time = datetime.timedelta(seconds = int(frame_number/input_video.fps - start_caption_offset))
						
						tmp_clip_start_time_str = (input_video.creation_time + relative_start_time).strftime("%Y%m%d_%H_%M_%S")
						output_filename = tmp_clip_start_time_str + '_' + inbox_entry.name
						output_filename = os.path.join(output_path, output_filename)
						input_video.export_video_part(output_filename, relative_start_time, datetime.timedelta(seconds = clip_duration))
						
						cv2.imshow('Frame',frame_cut)
						cv2.imshow('Mask',mask)
						cv2.waitKey(50)
					
					#print('Frame time:' + str((time.time()-float(frame_stopwatch_start))) + ' seconds')
		
		
		print('Elapsed time:' + str((time.time()-float(video_stopwatch_start))) + ' seconds')
		
		
print('Total elapsed time:' + str((time.time()-float(total_stopwatch_start))) + ' seconds')
