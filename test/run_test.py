import sys,os, logging

sys.path.append('..\code\Modules')
sys.path.append('..\code\Application')
import Application

logger = logging.getLogger(__name__)

tmp_video_folder = os.path.abspath("input_video")
tmp_user_folder  = os.path.abspath("users")

print("Starting test!\n")
logger.debug("------------------------Start------------------------------------\n")

app_instance = Application.Application()
app_instance.run(tmp_video_folder, tmp_user_folder)

print("Finished test!\n")