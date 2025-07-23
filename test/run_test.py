# Make main process create the new file and sub-processes append - probably not the nices way of doing this
import logging
import os

if __name__ == "__main__":
    write_mode = 'w'
else:
    write_mode = 'a'

logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode=write_mode)



if __name__ == "__main__":
    import sys, os, logging

    root_dir = os.path.join(os.path.dirname(__file__), '..')
    code_dir = os.path.join(root_dir, 'code')
    sys.path.append(os.path.join(code_dir, 'modules'))
    sys.path.append(os.path.join(code_dir, 'Application'))
    sys.path.append(os.path.join(code_dir, 'external_modules'))
    import Application
    from Enums import ActionOptions
    from ConfigurationHandler import ConfigurationHandler

    logger = logging.getLogger(__name__)


    #tmp_video_folder = os.path.abspath("F:/GoPro/Kabelpark/20180603")
    #tmp_video_folder = os.path.abspath("E:/DCIM/100GOPRO") #  - very slow to read from SD card using converter and build-in reader
    #tmp_user_folder  = os.path.abspath("F:/GoPro/BearVision/users")
    tmp_video_folder = os.path.abspath(os.path.join(root_dir, "test", "input_video"))
    tmp_user_folder  = os.path.abspath(os.path.join(root_dir, "test", "users"))
    tmp_config_file = os.path.abspath(os.path.join(root_dir, "test", "test_config.ini"))

    # list of actions to do in the test
    tmp_action_list = [ActionOptions.GENERATE_MOTION_FILES.value,
                       ActionOptions.INIT_USERS.value,
                       ActionOptions.MATCH_LOCATION_IN_MOTION_FILES.value,
                       ActionOptions.GENERATE_FULL_CLIP_OUTPUTS.value,
                       ActionOptions.GENERATE_TRACKER_CLIP_OUTPUTS.value]

    print("Starting test!\n")
    logger.debug("------------------------Start------------------------------------")

    tmp_options = ConfigurationHandler.read_config_file(tmp_config_file)
    app_instance = Application.Application()
    app_instance.run(tmp_video_folder, tmp_user_folder, tmp_action_list)

    print("\nFinished test!")
    logger.debug("-------------------------End-------------------------------------")