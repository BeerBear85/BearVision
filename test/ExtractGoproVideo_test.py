import logging


if __name__ == "__main__":
    write_mode = 'w'
else:
    write_mode = 'a'

logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode=write_mode)


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

    video_file_name_list = list()
    video_file_name_list.append('C:\\Users\\bjes\\OneDrive - MAN Energy Solutions SE\\personal\\BearVision\\test_video\\GP020511.MP4')

    #video_file_name_list = [video_file_name_list[0]] # single file test

    do_tracking = False
    visualize_tracking = False


    for video_file_name in video_file_name_list:

        if do_tracking:

            tracker = BearTracker()
            input_video_obj = GoproVideo()

            input_video_obj.init(video_file_name)
            tracker.init(video_file_name)

            while True:
                read_return_value, frame, frame_number = input_video_obj.read_frame()
                print(f'Frame number: {frame_number}')
                if read_return_value == 0:
                    print('Reached end of video')
                    break
                if read_return_value == 20: #GoPro error with empty frame
                    continue

                start_state = tracker.state
                tracker.calculate(frame)
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

