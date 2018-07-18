import sys,os, logging

sys.path.append('..\code\Modules')
sys.path.append('..\code\Application')
sys.path.append('..\code\external_modules')
import Application

logger = logging.getLogger(__name__)

#tmp_video_folder = os.path.abspath("input_video")
tmp_video_folder = os.path.abspath("F:/GoPro/Kabelpark/20180603")
#tmp_video_folder = os.path.abspath("E:/DCIM/100GOPRO") #  - very slow to read from SD card using converter and build-in reader
tmp_user_folder  = os.path.abspath("F:/GoPro/BearVision/users")

print("Starting test!\n")
logger.debug("------------------------Start------------------------------------")

app_instance = Application.Application()
app_instance.run(tmp_video_folder, tmp_user_folder)

print("\nFinished test!")
logger.debug("-------------------------End-------------------------------------")