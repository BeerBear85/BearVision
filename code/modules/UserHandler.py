import logging, os
import User

logger = logging.getLogger(__name__)


class UserHandler:
    def __init__(self):
        self.user_list = []
        return

    def init(self, arg_user_root_folder):
        logger.debug("Scanning of users in folder: " + arg_user_root_folder)
        user_root_folder_content = os.scandir(arg_user_root_folder)

        for user_folder in user_root_folder_content:
            if user_folder.is_dir():
                self.user_list.append( User.User(user_folder) )
        return
