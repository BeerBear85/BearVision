import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2

# Import pipeline components
from annotation_pipeline import (
    PipelineConfig,
    SamplingConfig,
    QualityConfig,
    YoloConfig,
    ExportConfig,
    VidIngest,
    QualityFilter,
    PreLabelYOLO,
    DatasetExporter,
)


def run_pipeline(video_path: str, output_dir: str) -> None:
    """Run the annotation pipeline while displaying detection results.

    Parameters
    ----------
    video_path: str
        Path to the input video.
    output_dir: str
        Directory where the dataset should be written.
    """
    cfg = PipelineConfig(
        videos=[video_path],
        sampling=SamplingConfig(step=1),
        quality=QualityConfig(blur=0, luma_min=0, luma_max=500),
        yolo=YoloConfig(weights="yolov8s.onnx", conf_thr=0.25),
        export=ExportConfig(output_dir=output_dir, format="yolo"),
    )

    ingest = VidIngest(cfg.videos, cfg.sampling)
    qf = QualityFilter(cfg.quality)
    yolo = PreLabelYOLO(cfg.yolo)
    exporter = DatasetExporter(cfg.export)

    for item in ingest:
        frame = item["frame"]
        if not qf.check(frame):
            continue
        boxes = yolo.detect(frame)
        for b in boxes:
            x1, y1, x2, y2 = map(int, b["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        exporter.save(item, boxes)

    exporter.close()
    cv2.destroyAllWindows()


class AnnotationGUI:
    """Minimal technical GUI for running the annotation pipeline."""

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Annotation Pipeline")

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        tk.Button(master, text="Select Video", command=self.select_video).pack(fill="x")
        tk.Label(master, textvariable=self.video_path, anchor="w").pack(fill="x")

        tk.Button(master, text="Select Output", command=self.select_output).pack(fill="x")
        tk.Label(master, textvariable=self.output_dir, anchor="w").pack(fill="x")

        tk.Button(master, text="Run", command=self.start).pack(fill="x")

    def select_video(self) -> None:
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")])
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
            messagebox.showerror("Missing paths", "Please select video and output directory")
            return

        thread = threading.Thread(target=run_pipeline, args=(video, output), daemon=True)
        thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    gui = AnnotationGUI(root)
    root.mainloop()
