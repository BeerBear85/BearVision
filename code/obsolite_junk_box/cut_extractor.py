
import argparse, cv2, time, os, datetime

from CutSpecification import CutSpecification
from GoproVideo import GoproVideo


#### Handle commandline arguments ####
parser = argparse.ArgumentParser(description='Extract moving box from HD video')
parser.add_argument('--file', default='bboxes_136.txt', help='Specify the filename of the clip specfile from MATLAB script')
args = parser.parse_args()
#specfile_name = args.file

#Program start initilisation
cut_spec_inbox_path = './cut_spec_inbox'
output_path = './outputs'

output_codex = cv2.VideoWriter_fourcc(*'DIVX')
output_fps = 15
input_video = GoproVideo()

while (1):
	#if cv2.waitKey(1000) & 0xFF == ord('q'): #1 Hz
	#	break
	time.sleep(1.0) 
	inbox_list = os.listdir(cut_spec_inbox_path)
	
	if inbox_list: #list is not empty
		time.sleep(0.2) #make sure matlab has finished writing
		stopwatch_start = time.time()
		specfile_name = os.path.join(cut_spec_inbox_path,inbox_list[0])
		print('Extracting cut specification: ', specfile_name)
		
		### Initilise cut specific values - part 1 ###
		specfile = CutSpecification(specfile_name) #Parse the specfile
		input_video.init(specfile.video_filename)
		
		### Initilise cut specific values - part 2 ###
		# ///@todo Video clip generation should be move to own class or function
		#Generate output file
		(dummy, tmp_rel_video_filename) = os.path.split(os.path.splitext(specfile.video_filename)[0])
		tmp_clip_start_time_str = (input_video.creation_time + datetime.timedelta(seconds = specfile.start_frame/input_video.fps)).strftime("%Y%m%d_%H_%M_%S")
		output_filename = tmp_clip_start_time_str + '_' + tmp_rel_video_filename + '_' + str(specfile.track_id)
		output_filename = os.path.join(output_path, output_filename)
		
		input_video.set_start_point(specfile.start_frame) #Set start point of video
		writer_object = cv2.VideoWriter(output_filename + '.avi', output_codex, output_fps, (specfile.box_dimensions[0], specfile.box_dimensions[1]))
		
		### Read frames and write cut_out ###
		for relative_frame_number in range(0, specfile.track_age):
			#print("Extracting frame: " + str(relative_frame_number))
			read_return_value, frame = input_video.read_frame()
			if (read_return_value == 0): #end of file
				break
			#print("Extracted frame: " + str(relative_frame_number))
			
			x_start = specfile.box_coordinates[relative_frame_number][0]
			y_start = specfile.box_coordinates[relative_frame_number][1]
			x_end   = x_start + specfile.box_dimensions[0]	
			y_end   = y_start + specfile.box_dimensions[1]
			
			#Range checks:  
			if x_start < 0:
				x_end   = (x_end - x_start) #Note that x_start is negative
				x_start = int(0)
				
			if y_start < 0:
				y_end   = (y_end - y_start) #Note that x_start is negative
				y_start = int(0)
			
			if x_end >= input_video.width:
				x_start = x_start - (x_end - input_video.width) - 1
				x_end   = input_video.width - 1
				
			if y_end >= input_video.height:
				y_start = y_start - (y_end - input_video.height) - 1
				y_end   = input_video.height - 1
			
			frame = frame[int(y_start):int(y_end), int(x_start):int(x_end)] #Rows before colunms
			#print("Cropped frame: " + str(relative_frame_number))
			writer_object.write(frame)
			#print("Wrote frame: " + str(relative_frame_number))
			
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		
		### Finish/clean up after cut extraction ###
		writer_object.release()
		os.rename(specfile_name, (output_filename + '.txt')) #move spec file
		
		print('Finished writing video: ', output_filename)
		duration = (time.time()-float(stopwatch_start))
		print('Elapsed time:' + str(duration) + ' seconds')
		
