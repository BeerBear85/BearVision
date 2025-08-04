import logging, os
import User
from Enums import ClipTypes

logger = logging.getLogger(__name__)


class UserHandler:
    """Manage multiple :class:`User` instances and provide helper methods."""

    def __init__(self):
        """Initialise an empty handler."""
        self.user_list = []
        return

    def init(self, arg_user_root_folder):
        """Populate the handler by scanning a directory for users.

        Args:
            arg_user_root_folder (str): Path containing user subdirectories.

        Returns:
            None
        """
        logger.debug("Scanning of users in folder: " + arg_user_root_folder)
        user_root_folder_content = os.scandir(arg_user_root_folder)

        for user_folder in user_root_folder_content:
            if user_folder.is_dir():
                # Only directories are considered valid users because a file
                # would not contain the expected structure.
                self.user_list.append(User.User(user_folder))
        return

    def find_valid_user_match(self, target_date, target_location):
        """Find a user whose recorded data matches the given time and place.

        Args:
            target_date (datetime.datetime): Timestamp to match against user
                data.
            target_location (Iterable[float]): Latitude/longitude pair.

        Returns:
            User or int: The matching :class:`User` instance or ``0`` if none
            is found. Returning ``0`` instead of ``None`` preserves historical
            behaviour of the caller.
        """
        user_match = 0
        logger.debug(
            "Looking for users at time: "
            + target_date.strftime("%Y%m%d_%H_%M_%S_%f")
            + " near location:"
            + str(target_location)
        )
        for user in self.user_list:
            if user.is_close(target_date, target_location):
                user_match = user
                logger.debug(
                    "User match found at time: "
                    + target_date.strftime("%Y%m%d_%H_%M_%S_%f")
                    + " near location:"
                    + str(target_location)
                )
        return user_match

    def create_clip_specifications(self, clip_type: ClipTypes):
        """Aggregate clip specifications from all users.

        Args:
            clip_type (ClipTypes): Type of clip to generate for each match.

        Returns:
            list: List of :class:`BasicClipSpecification` objects.
        """
        logger.debug("Creating specification objects for the found matches")
        list_of_clip_specs = []
        for user in self.user_list:
            user_clip_list = user.create_clip_specifications(clip_type)
            list_of_clip_specs.extend(user_clip_list)
        return list_of_clip_specs

    def filter_obstacle_matches(self, arg_user_match_minimum_interval: float):
        """Remove user matches that are too close in time to each other.

        Args:
            arg_user_match_minimum_interval (float): Minimum allowed interval
                between matches in seconds.

        Returns:
            None
        """
        for user in self.user_list:
            user.filter_obstacle_matches(arg_user_match_minimum_interval)
        return
