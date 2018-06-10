import sys, logging, tkinter

sys.path.append('code\Modules')
sys.path.append('code\Application')
sys.path.append('code\external_modules')

import Application
from GUI import BearVisionGUI

logger = logging.getLogger(__name__)


logger.debug("------------------------Start------------------------------------")

app_instance = Application.Application()

#Start GUI
GUI_root = tkinter.Tk()
my_gui = BearVisionGUI(GUI_root, app_instance)
GUI_root.mainloop()


#app_instance.run(tmp_video_folder, tmp_user_folder)

logger.debug("-------------------------End-------------------------------------")