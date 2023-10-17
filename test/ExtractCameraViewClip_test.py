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

    from ExtractCameraViewClip import ExtractCameraViewClip

    logger = logging.getLogger(__name__)

    pickle_file_name = os.path.abspath("test/test_video/TestMovie1_113_tracking.pkl")

    my_extracter = ExtractCameraViewClip()
    my_extracter.init(pickle_file_name)
    my_extracter.run()


