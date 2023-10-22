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

    for video_file_name in video_file_name_list:

        tracker = BearTracker()
        my_extracter = ExtractCameraViewClip()
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

            tracker.calculate(frame)

            if tracker.state == State.DONE:
                print(f'Tracking completed at frame {frame_number}')
                pickle_file_name = tracker.save_data()
                my_extracter.init(pickle_file_name)
                my_extracter.run()
                tracker.reset()
                print(f'Extract complete for {pickle_file_name}')
