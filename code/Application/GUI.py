from tkinter import *
from tkinter import filedialog
import os, logging

logger = logging.getLogger(__name__)

pa_default_input_video_folder = "test/input_video"
pa_default_user_folder = "test/users"


class BearVisionGUI:
    def __init__(self, arg_master, arg_app_ref):
        self.master = arg_master
        self.app_ref = arg_app_ref
        self.master.title("BearVision WakeVision")
        self.master.geometry("500x500")

        self.welcome_label = Label(self.master, text="BearVision - WakeVison", bg='red', font=('Helvetica', '20'))
        self.welcome_label.pack(fill=X, side=TOP)

        self.folder_selection_frame = Frame(self.master)
        self.folder_selection_frame.pack(fill=X, pady=10, side=TOP)
        self.folder_selection_frame.columnconfigure(0, weight=3)
        self.folder_selection_frame.columnconfigure(1, weight=1)

        self.video_folder_text = StringVar()
        self.video_folder_text.set( os.path.abspath(pa_default_input_video_folder) )
        self.video_folder_entry = Entry(self.folder_selection_frame, textvariable=self.video_folder_text)
        self.video_folder_entry.grid(row=0, column=0, sticky=W+E)
        self.video_folder_button = Button(self.folder_selection_frame, text="Select input video folder", command=self.set_input_video_folder)
        self.video_folder_button.grid(row=0, column=1, sticky=W+E)

        self.user_folder_text = StringVar()
        self.user_folder_text.set( os.path.abspath(pa_default_user_folder) )
        self.user_folder_entry = Entry(self.folder_selection_frame, textvariable=self.user_folder_text, width=60)
        self.user_folder_entry.grid(row=1, column=0, sticky=W+E)
        self.user_folder_button = Button(self.folder_selection_frame, text="Select user base folder", command=self.set_user_folder)
        self.user_folder_button.grid(row=1, column=1, sticky=W+E)

        self.run_options = Listbox(self.master, selectmode=MULTIPLE )
        self.run_options.insert(END, "Generate motion start files")
        self.run_options.insert(END, "Initialize users")
        self.run_options.insert(END, "Match user locations to motion files")
        self.run_options.insert(END, "Generate output videos")
        #  self.run_options.pack(fill=X, pady=10)
        self.run_options.selection_set(0,self.run_options.size())  # select all options

        self.run_button = Button(self.master, text="Run", command= self.run, bg='green3', height = 1, width = 10, font=('Helvetica', '20'))
        self.run_button.pack(pady=10)

    def set_input_video_folder(self):
        selected_directory = filedialog.askdirectory()
        self.video_folder_text.set(selected_directory)
        logger.info("Setting input video folder to: " + selected_directory)

    def set_user_folder(self):
        selected_directory = filedialog.askdirectory()
        self.user_folder_text.set(selected_directory)
        logger.info("Setting user folder to: " + selected_directory)

    def run(self):
        logger.debug("run()")
        self.app_ref.run(self.video_folder_text.get(), self.user_folder_text.get())
        #print("Running selections: " + str(self.run_options.curselection() ))


#GUI_root = Tk()
#my_gui = BearVisionGUI(GUI_root)
#GUI_root.mainloop()
