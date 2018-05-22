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

    def find_valid_user_match(self,  target_date, target_location):
        user_match = 0
        logger.debug("Looking for users at time: " + target_date.strftime("%Y%m%d_%H_%M_%S") + " near location:" + str(target_location))
        for user in self.user_list:
            if user.is_close(target_date, target_location):
                user_match = user
                logger.debug("User match found at time: " + target_date.strftime("%Y%m%d_%H_%M_%S") + " near location:" + str(target_location))
        return user_match

    def create_full_clip_specifications(self):
        logger.debug("Creating specification objects for the found matches")
        full_list_of_clip_specs = []
        for user in self.user_list:
            user_clip_list = user.create_full_clip_specifications()
            full_list_of_clip_specs.extend(user_clip_list)
        return full_list_of_clip_specs