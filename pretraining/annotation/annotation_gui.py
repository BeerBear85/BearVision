import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import annotation_pipeline as ap


def run_pipeline(video_path: str, output_dir: str) -> None:
    """Run the annotation pipeline and export a labeled dataset."""

    cfg = ap.PipelineConfig(
        videos=[video_path],
        sampling=ap.SamplingConfig(step=1),
        quality=ap.QualityConfig(blur=0, luma_min=0, luma_max=500),
        yolo=ap.YoloConfig(weights="yolov8s.onnx", conf_thr=0.25),
        export=ap.ExportConfig(output_dir=output_dir),
    )
    ap.run(cfg, show_preview=True)


class AnnotationGUI:
    """Minimal technical GUI for running the annotation pipeline."""

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Annotation Pipeline")

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        tk.Button(
            master, text="Select Video", command=self.select_video
        ).pack(fill="x")
        tk.Label(
            master, textvariable=self.video_path, anchor="w"
        ).pack(fill="x")

        tk.Button(
            master, text="Select Output", command=self.select_output
        ).pack(fill="x")
        tk.Label(
            master, textvariable=self.output_dir, anchor="w"
        ).pack(fill="x")

        tk.Button(master, text="Run", command=self.start).pack(fill="x")

    def select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)

    def select_output(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def start(self) -> None:
        video = self.video_path.get()
        output = self.output_dir.get()
        if not video or not output:
            messagebox.showerror(
                "Missing paths", "Please select video and output directory"
            )
            return

        thread = threading.Thread(
            target=run_pipeline, args=(video, output), daemon=True
        )
        thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    gui = AnnotationGUI(root)
    root.mainloop()
