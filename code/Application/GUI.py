"""Graphical user interface for configuring and running BearVision."""

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import os, logging
from ConfigurationHandler import ConfigurationHandler
from Enums import ActionOptions
from MotionROISelector import MotionROISelector

logger = logging.getLogger(__name__)

# The much better way is to use ENUMS as:
commands = {
    ActionOptions.GENERATE_MOTION_FILES: "Generate motion start files",
    ActionOptions.INIT_USERS: "Initialize users",
    ActionOptions.MATCH_LOCATION_IN_MOTION_FILES: "Match user locations to motion files",
    ActionOptions.GENERATE_FULL_CLIP_OUTPUTS: "Generate full clip output videos",
    ActionOptions.GENERATE_TRACKER_CLIP_OUTPUTS: "Generate tracker clip output videos",
}


class BearVisionGUI:
    """Tkinter based front end for selecting and running processing steps."""

    def __init__(self, arg_master: Tk, arg_app_ref) -> None:
        """Create the GUI and populate it with widgets.

        Parameters
        ----------
        arg_master : Tk
            Root window of the application.
        arg_app_ref : Application
            Reference to the core application used to execute actions.
        """
        tmp_options = ConfigurationHandler.get_configuration()
        self.master = arg_master
        self.app_ref = arg_app_ref
        self.master.title("BearVision - WakeVision")
        self.master.geometry("500x500")

        self.welcome_label = Label(self.master, text="BearVision - WakeVison", bg="red", font=("Helvetica", "20"))
        self.welcome_label.pack(fill=X, side=TOP)

        # The folder frame groups both input and user folder selectors for a clean layout.
        self.folder_selection_frame = Frame(self.master)
        self.folder_selection_frame.pack(fill=X, pady=10, side=TOP)
        self.folder_selection_frame.columnconfigure(0, weight=3)
        self.folder_selection_frame.columnconfigure(1, weight=1)

        self.video_folder_text = StringVar()
        if tmp_options is not None:
            # Pre-populate the entry so the user can accept the previous path quickly.
            self.video_folder_text.set(os.path.abspath(tmp_options["GUI"]["video_path"]))
        self.video_folder_entry = Entry(self.folder_selection_frame, textvariable=self.video_folder_text)
        self.video_folder_entry.grid(row=0, column=0, sticky=W + E)
        self.video_folder_button = Button(
            self.folder_selection_frame,
            text="Select input video folder",
            command=self.set_input_video_folder,
        )
        self.video_folder_button.grid(row=0, column=1, sticky=W + E)

        self.user_folder_text = StringVar()
        if tmp_options is not None:
            self.user_folder_text.set(os.path.abspath(tmp_options["GUI"]["user_path"]))
        self.user_folder_entry = Entry(self.folder_selection_frame, textvariable=self.user_folder_text, width=60)
        self.user_folder_entry.grid(row=1, column=0, sticky=W + E)
        self.user_folder_button = Button(
            self.folder_selection_frame,
            text="Select user base folder",
            command=self.set_user_folder,
        )
        self.user_folder_button.grid(row=1, column=1, sticky=W + E)

        self.run_options = Listbox(self.master, selectmode=MULTIPLE)
        for alias in commands:
            self.run_options.insert(END, commands[alias])

        self.run_options.pack(fill=X, pady=10)
        self.run_options.selection_set(0, self.run_options.size())  # select all options

        # Frame and button for motion ROI selection
        self.motion_ROI_selection_frame = Frame(self.master)
        self.motion_ROI_selection_frame.pack(fill=X, pady=10, side=TOP)
        self.motion_ROI_selection_frame.columnconfigure(0, weight=3)
        self.motion_ROI_selection_frame.columnconfigure(1, weight=1)

        self.motion_ROI_text = StringVar()
        if tmp_options is not None:
            self.motion_ROI_text.set(tmp_options["MOTION_DETECTION"]["search_box_dimensions"])
        self.video_folder_entry = Entry(self.motion_ROI_selection_frame, textvariable=self.motion_ROI_text)
        self.video_folder_entry.grid(row=0, column=0, sticky=W + E)
        self.video_folder_button = Button(
            self.motion_ROI_selection_frame,
            text="Select motion detection ROI",
            command=self.set_motion_detection_ROI,
        )
        self.video_folder_button.grid(row=0, column=1, sticky=W + E)

        # Create button frame
        self.button_frame = Frame(self.master)
        self.button_frame.pack(fill=X, pady=10)

        self.config_load_button = Button(
            self.button_frame,
            text="Load Config",
            command=self.load_config,
            bg="green3",
            height=1,
            width=10,
            font=("Helvetica", "20"),
        )
        self.config_load_button.pack(side=LEFT)

        self.run_button = Button(
            self.button_frame,
            text="Run",
            command=self.run,
            bg="green3",
            height=1,
            width=10,
            font=("Helvetica", "20"),
        )
        self.run_button.pack(side=RIGHT, pady=10)

        self.status_label_text = StringVar()
        self.status_label_text.set("Ready")
        if tmp_options is None:
            self.status_label_text.set("No parameters")
        self.status_label = Label(self.master, textvariable=self.status_label_text, bg="yellow", font=("Helvetica", "20"))
        self.status_label.pack(fill=X, side=BOTTOM, pady=10)

    def set_input_video_folder(self, arg_directory_path: str | None = None) -> None:
        """Ask the user for a video folder and store the choice.

        Parameters
        ----------
        arg_directory_path : str | None, optional
            Preselected directory. If ``None`` a dialog is shown.
        """
        if arg_directory_path is None:
            arg_directory_path = filedialog.askdirectory(initialdir=self.video_folder_text.get())
        self.video_folder_text.set(os.path.abspath(arg_directory_path))
        logger.info("Setting input video folder to: %s", arg_directory_path)

    def set_user_folder(self, arg_directory_path: str | None = None) -> None:
        """Ask the user for the user base folder and store it.

        Parameters
        ----------
        arg_directory_path : str | None, optional
            Preselected directory. If ``None`` a dialog is shown.
        """
        if arg_directory_path is None:
            arg_directory_path = filedialog.askdirectory(initialdir=self.user_folder_text.get())
        self.user_folder_text.set(os.path.abspath(arg_directory_path))
        logger.info("Setting user folder to: %s", arg_directory_path)

    def set_motion_detection_ROI(self) -> None:
        """Launch ROI selector and update the configuration with the chosen region."""

        tempROISelector = MotionROISelector()
        tmpROI = tempROISelector.SelectROI(self.video_folder_text.get())
        tmp_options = ConfigurationHandler.get_configuration()
        self.motion_ROI_text.set(tmp_options["MOTION_DETECTION"]["search_box_dimensions"])
        logger.info("Setting motion ROI to: %s", tmpROI)

    def run(self) -> None:
        """Trigger the application to run selected actions."""
        logger.debug("run()")
        tmp_options = ConfigurationHandler.get_configuration()
        if tmp_options is None:
            self.status_label_text.set("No parameters")
            return
        self.status_label_text.set("Busy")
        self.status_label.update()
        # Pass GUI selections to the application; using indices keeps coupling loose.
        self.app_ref.run(self.video_folder_text.get(), self.user_folder_text.get(), self.run_options.curselection())
        self.status_label_text.set("Ready")

    def load_config(self) -> None:
        """Load a configuration file and update GUI fields."""
        logger.debug("load_config()")
        tmp_config_file = filedialog.askopenfilename(initialdir=ConfigurationHandler.get_configuration_path())
        tmp_options = ConfigurationHandler.read_config_file(tmp_config_file)

        # update file selection boxes and GUI
        self.set_input_video_folder(tmp_options["GUI"]["video_path"])
        self.set_user_folder(tmp_options["GUI"]["user_path"])
        self.status_label_text.set("Ready")






#GUI_root = Tk()
#my_gui = BearVisionGUI(GUI_root)
#GUI_root.mainloop()
