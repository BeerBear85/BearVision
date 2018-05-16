import sys,os

sys.path.append('..\code\Modules')
sys.path.append('..\code\Application')
import Application

tmp_video_folder = os.path.abspath("input_video")
tmp_user_folder  = os.path.abspath("users")

print("Starting test!\n")

app_instance = Application.Application()
app_instance.run(tmp_video_folder, tmp_user_folder)

print("Finished test!\n")