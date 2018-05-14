import logging

logging.basicConfig(filename='debug.log',level=logging.DEBUG)

class Application:
    def __init__(self):
        logging.debug("appliation created\n")

    def init(self):
        logging.debug("Init called\n")

    def run(self, arg_input_video_folder, arg_user_folder):
        logging.debug("Running Application with video folder: " + arg_input_video_folder + " user folder: " + arg_user_folder + "\n")