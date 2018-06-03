from tkinter import *
from tkinter import filedialog
from tkinter import font

pa_default_input_video_folder = "input_video"
pa_default_user_folder = "users"

#big_font = font.Font(family='Helvetica', size=36, weight='bold')
#appHighlightFont = font.Font(family='Helvetica', size=12, weight='bold')
#font.families()

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("BearVision WakeVision")
        master.geometry("500x500")

        self.welcome_label = Label(master, text="Welcome to WakeVison")
        self.welcome_label.pack(side=TOP)

        self.video_folder_frame = Frame(master)
        self.video_folder_frame.pack()
        self.video_folder_text = StringVar()
        self.video_folder_text.set(pa_default_input_video_folder)
        self.video_folder_label = Label(self.video_folder_frame, textvariable=self.video_folder_text)
        self.video_folder_label.pack(side=LEFT)
        self.video_folder_button = Button(self.video_folder_frame, text="Select input video folder", command=self.set_input_video_folder)
        self.video_folder_button.pack(side=LEFT)

        self.user_folder_frame = Frame(master)
        self.user_folder_frame.pack()
        self.user_folder_text = StringVar()
        self.user_folder_text.set(pa_default_user_folder)
        self.user_folder_label = Label(self.user_folder_frame, textvariable=self.user_folder_text)
        self.user_folder_label.pack( side=LEFT )
        self.user_folder_button = Button(self.user_folder_frame, text="Select user base folder", command=self.set_user_folder)
        self.user_folder_button.pack( side=LEFT )

        #self.check_var_1 = IntVar()
        #self.checkbox_1 = Checkbutton(master, text="Option 1", variable = self.check_var_1)
        #self.checkbox_1.pack()

        self.run_options = Listbox(master, selectmode=MULTIPLE )
        self.run_options.insert(1, "Generate motion start files")
        self.run_options.insert(2, "Initialize users")
        self.run_options.insert(3, "Match user locations to motion files")
        self.run_options.insert(4, "Generate output videos")
        self.run_options.pack()
        #self.run_options.activate(1)
        #self.run_options.activate(self.run_options.index(3))

        self.run_button = Button(master, text="Run", command=self.run)
        self.run_button.pack()

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
        print("Running checkbox_1: " + str(self.check_var_1.get()))



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
