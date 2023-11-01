# pylint: disable=E0401
import logging


if __name__ == "__main__":
    write_mode = 'w'
else:
    write_mode = 'a'

logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode=write_mode)

# Create a console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler) # Add the console handler to the root logger


if __name__ == "__main__":
    import sys
    import os
    import cv2

    modules_abs_path = os.path.abspath("code/modules")
    sys.path.append(modules_abs_path)

    from BearTracker import BearTracker
    from ExtractCameraViewClip import ExtractCameraViewClip
    from GoproVideo import GoproVideo
    from BearTracker import State

    logger = logging.getLogger(__name__)

    one_drive_folder = os.path.join('C:','Users','bjes','OneDrive - MAN Energy Solutions SE','personal','BearVision','test_video')
    

    video_file_name_list = list()
    #video_file_name_list.append(os.path.join(one_drive_folder,'GP020511','GP020511.MP4'))
    video_file_name_list.append(os.path.join(one_drive_folder,'GP010554','GP010554.MP4'))

    #video_file_name_list = [video_file_name_list[0]] # single file test

    do_tracking = True
    visualize_tracking = False




    for video_file_name in video_file_name_list:

        logger.info("Processing %s", video_file_name)


        #Delete all existing pickle files in folder
        folder_path = os.path.dirname(video_file_name_list[0]) #Get folder of video_file_name
        pickle_file_name_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
        for pickle_file_name in pickle_file_name_list:
            os.remove(pickle_file_name)

        if do_tracking:

            tracker = BearTracker()
            input_video_obj = GoproVideo()

            input_video_obj.init(video_file_name)
            tracker.init(video_file_name, input_video_obj.fps)

            while True:
                read_return_value, frame, frame_number = input_video_obj.read_frame()
                if frame_number % 20 == 0:
                    print(f'Frame number: {frame_number}')
                if read_return_value == 0:
                    print('Reached end of video')
                    break
                if read_return_value == 20: #GoPro error with empty frame (will skip 5 frames)
                    continue

                start_state = tracker.state
                tracker.calculate(frame, frame_number)
                if visualize_tracking and start_state == State.TRACKING:
                    frame = tracker.visualize_state(frame)

                    #Scale frame to 50% for better overview
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('frame', frame)
                    #Wait for 1 ms for keypress
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        ## Extract camera view clip ##
        video_folder = os.path.dirname(video_file_name) #Get folder of video_file_name
        my_extracter = ExtractCameraViewClip()
        my_extracter.init()
        my_extracter.extract_folder(video_folder)

