
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
    import sys, tkinter

    sys.path.append('code\Modules')
    sys.path.append('code\Application')
    sys.path.append('code\external_modules')

    from Application import Application
    from GUI import BearVisionGUI
    from ConfigurationHandler import ConfigurationHandler

    logger = logging.getLogger(__name__)

    logger.debug("------------------------Start------------------------------------")

    ConfigurationHandler.read_last_used_config_file()
    app_instance = Application()

    #Start GUI
    GUI_root = tkinter.Tk()
    my_gui = BearVisionGUI(GUI_root, app_instance)
    GUI_root.mainloop()


    logger.debug("-------------------------End-------------------------------------")