from tkinter import *
from tkinter import filedialog
from tkinter import font
import os

pa_default_input_video_folder = "input_video"
pa_default_user_folder = "users"

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("BearVision WakeVision")
        master.geometry("500x500")

        self.welcome_label = Label(master, text="BearVision - WakeVison", bg='red', font=('Helvetica', '20'))
        self.welcome_label.pack(fill=X, side=TOP)

        self.folder_selection_frame = Frame(master)
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

        self.run_options = Listbox(master, selectmode=MULTIPLE )
        self.run_options.insert(END, "Generate motion start files")
        self.run_options.insert(END, "Initialize users")
        self.run_options.insert(END, "Match user locations to motion files")
        self.run_options.insert(END, "Generate output videos")
        self.run_options.pack(fill=X, pady=10)
        self.run_options.selection_set(0,self.run_options.size()) #select all

        self.run_button = Button(master, text="Run", command=self.run, bg='green3', height = 1, width = 10, font=('Helvetica', '20'))
        self.run_button.pack(pady=10)

    def set_input_video_folder(self):
        selected_directory = filedialog.askdirectory()
        self.video_folder_text.set(selected_directory)
        print("Setting input video folder to: " + selected_directory)

    def set_user_folder(self):
        selected_directory = filedialog.askdirectory()
        self.user_folder_text.set(selected_directory)
        print("Setting user folder to: " + selected_directory)

    def run(self):
        print("Running!")
        print("Running selections: " + str(self.run_options.curselection() ))



root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()

#

#def run(self, arg_input_video_folder, arg_user_root_folder):
#    logger.info(
#        "Running Application with video folder: " + arg_input_video_folder + " user folder: " + arg_user_root_folder + "\n")
#   if not os.path.exists(arg_input_video_folder):
#        raise ValueError("Video folder is not a valid folder: " + arg_input_video_folder)
#    if not os.path.exists(arg_user_root_folder):
#        raise ValueError("User folder is not a valid folder: " + arg_user_root_folder)


#    self.motion_start_detector.create_motion_start_files(arg_input_video_folder)

#    self.user_handler.init(arg_user_root_folder)
#    self.motion_time_user_matching.match_motion_start_times_with_users(arg_input_video_folder, self.user_handler)
#    clip_specification_list = self.user_handler.create_full_clip_specifications()

#    self.full_clip_cut_extractor.extract_full_clip_specifications(clip_specification_list)
