from enum import Enum, unique

@unique
class ActionOptions(Enum):
    GENERATE_MOTION_FILES = 0
    INIT_USERS = 1
    MATCH_LOCATION_IN_MOTION_FILES = 2
    GENERATE_FULL_CLIP_OUTPUTS = 3
    GENERATE_TRACKER_CLIP_OUTPUTS = 4
    
@unique
class ClipTypes(Enum):
    FULL_CLIP = 1
    TRACKER_CLIP = 2
