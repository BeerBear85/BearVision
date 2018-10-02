# Make main process create the new file and sub-processes append - probably not the nices way of doing this
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
    import sys, os, logging

    sys.path.append('..\code\Modules')
    sys.path.append('..\code\Application')
    sys.path.append('..\code\external_modules')
    import Application
    from Enums import ActionOptions
    from ConfigurationHandler import ConfigurationHandler

    logger = logging.getLogger(__name__)


    #tmp_video_folder = os.path.abspath("F:/GoPro/Kabelpark/20180603")
    #tmp_video_folder = os.path.abspath("E:/DCIM/100GOPRO") #  - very slow to read from SD card using converter and build-in reader
    #tmp_user_folder  = os.path.abspath("F:/GoPro/BearVision/users")
    tmp_video_folder = os.path.abspath("input_video")
    tmp_user_folder  = os.path.abspath("users")
    tmp_config_file = os.path.abspath("test_config.ini")

    tmp_action_list = [ActionOptions.INIT_USERS.value, ActionOptions.MATCH_LOCATION_IN_MOTION_FILES.value]

    print("Starting test!\n")
    logger.debug("------------------------Start------------------------------------")

    tmp_options = ConfigurationHandler.read_config_file(tmp_config_file)
    app_instance = Application.Application()
    app_instance.run(tmp_video_folder, tmp_user_folder, tmp_action_list)

    print("\nFinished test!")
    logger.debug("-------------------------End-------------------------------------")