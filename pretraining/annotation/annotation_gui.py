import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import copy

import cv2
import numpy as np
from typing import Callable

import annotation_pipeline as ap


def run_pipeline(
    cfg: ap.PipelineConfig,
    video_path: str,
    output_dir: str,
    frame_callback: Callable[[np.ndarray], None],
) -> None:
    """Execute the pipeline for one video and export a dataset.

    Purpose
    -------
    Reuse a base configuration and invoke :func:`annotation_pipeline.run` so
    that frames, labels and interpolated trajectories are generated for the
    chosen video while emitting per-frame previews.

    Inputs
    ------
    cfg: ap.PipelineConfig
        Base configuration object which will be copied to avoid mutating
        caller state.
    video_path: str
        Path to the source video file.
    output_dir: str
        Directory where the dataset will be written.
    frame_callback: Callable[[np.ndarray], None]
        Function receiving each processed frame for preview rendering.

    Outputs
    -------
    None
        Files are written to ``output_dir`` and ``frame_callback`` is invoked
        for every exported frame.
    """

    # Reset progress so a new run starts with a clean status snapshot.
    ap.status = ap.PipelineStatus()
    cfg = copy.deepcopy(cfg)
    cfg.videos = [video_path]
    cfg.export.output_dir = output_dir
    # The callback feeds the GUI a copy of each frame.  We set
    # ``show_preview`` to ``False`` because the GUI now controls preview
    # rendering in its own OpenCV window rather than relying on the pipeline's
    # post-run display.
    ap.run(cfg, show_preview=False, frame_callback=frame_callback)


class AnnotationGUI:
    """Simple Tkinter front-end for the annotation pipeline."""

    def __init__(self, master: tk.Tk):
        """Create widgets and bind actions.

        Parameters
        ----------
        master: tk.Tk
            Root Tk instance used to place widgets.

        Outputs
        -------
        None
            Widgets are attached to ``master`` and await user interaction.
        """
        self.master = master
        master.title("Annotation Pipeline")

        # Load configuration so preview scaling can be adjusted via YAML without
        # touching the code. Path resolution stays relative to this file.
        cfg_path = Path(__file__).with_name("sample_config.yaml")
        self.base_cfg = ap._ensure_cfg(str(cfg_path))
        self.preview_scaling = self.base_cfg.preview_scaling
        ap.status = ap.PipelineStatus()  # Reset status so GUI starts in "Idle".

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status = tk.StringVar(value="Idle")
        # A separate variable tracks frame progress so that the textual
        # description of the current step remains uncluttered by numbers.
        self.frame_progress = tk.StringVar(value="Frame: 0/0")

        tk.Button(master, text="Select Video", command=self.select_video).pack(fill="x")
        tk.Label(master, textvariable=self.video_path, anchor="w").pack(fill="x")

        tk.Button(master, text="Select Output", command=self.select_output).pack(fill="x")
        tk.Label(master, textvariable=self.output_dir, anchor="w").pack(fill="x")

        # Keep a reference to the Run button so we can disable it during
        # processing, preventing duplicate launches.
        self.run_btn = tk.Button(master, text="Run", command=self.start)
        self.run_btn.pack(fill="x")
        tk.Label(master, textvariable=self.status, anchor="w").pack(fill="x")
        # Present the numeric progress in its own label so long jobs can be
        # monitored at a glance without parsing additional status text.
        tk.Label(master, textvariable=self.frame_progress, anchor="w").pack(fill="x")

        # The GUI used to embed previews directly in the window, but on some
        # platforms that resulted in blank output.  Instead we delegate frame
        # display to an OpenCV window created on demand, yielding more reliable
        # cross-platform behaviour while keeping the Tk widgets minimal.
        self._preview_window: str | None = None

        # Double the window's dimensions for better visibility without the
        # caller needing to guess an appropriate size ahead of widget creation.
        master.update_idletasks()
        w, h = master.winfo_width(), master.winfo_height()
        master.geometry(f"{w*2}x{h*2}")
        self.refresh_status()
    def select_video(self) -> None:
        """Prompt the user to select a video file.

        Purpose
        -------
        Allow users to browse for an input file without typing paths.

        Inputs
        ------
        None
            Uses a file dialog instead of parameters.

        Outputs
        -------
        None
            The chosen path is stored in ``self.video_path``.
        """
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
        """Prompt the user to select an output directory.

        Purpose
        -------
        Collect the dataset destination from the user through a GUI dialog.

        Inputs
        ------
        None
            Destination is chosen interactively.

        Outputs
        -------
        None
            The path is stored in ``self.output_dir``.
        """
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir.set(path)

    def start(self) -> None:
        """Launch the pipeline in a background thread.

        Purpose
        -------
        Kick off processing without freezing the GUI.

        Inputs
        ------
        None
            Uses previously selected paths stored on the instance.

        Outputs
        -------
        None
            A daemon thread is spawned to run the pipeline.
        """
        video = self.video_path.get()
        output = self.output_dir.get()
        if not video or not output:
            messagebox.showerror(
                "Missing paths", "Please select video and output directory"
            )
            return
        # Disable the button immediately so accidental double-clicks do not
        # spawn multiple pipeline instances competing for resources.
        self.run_btn.config(state=tk.DISABLED)
        # Running in a thread keeps the GUI responsive during processing.
        thread = threading.Thread(
            target=self._run_pipeline_thread, args=(video, output), daemon=True
        )
        thread.start()

    def _run_pipeline_thread(self, video: str, output: str) -> None:
        """Execute the pipeline and re-enable the Run button when done.

        Purpose
        -------
        Isolate long-running work from the GUI thread while ensuring the
        interface becomes usable again once processing completes.

        Inputs
        ------
        video: str
            Path to the video selected by the user.
        output: str
            Destination directory for generated annotations.

        Outputs
        -------
        None
            After completion the Run button is re-enabled on the main thread.
            Any exceptions are surfaced via a message box for visibility.
        """
        try:
            run_pipeline(self.base_cfg, video, output, self.on_frame)
        except Exception as exc:  # pragma: no cover - GUI presentation
            # Without this handler users receive no feedback when the pipeline
            # fails (e.g. due to missing video/weights).  Showing a message box
            # makes the failure explicit and keeps the GUI responsive.
            self.master.after(0, lambda: messagebox.showerror("Pipeline error", str(exc)))
        finally:
            # ``after`` hands control back to the Tk main loop so widget state is
            # manipulated safely from the GUI thread. We also ensure any preview
            # window is closed so subsequent runs start cleanly.
            self.master.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.master.after(0, self._close_preview_window)

    def on_frame(self, frame: np.ndarray) -> None:
        """Schedule display of a processed frame on the main thread.

        Purpose
        -------
        The pipeline emits frames from a worker thread. Tkinter widgets must be
        updated on the main thread, so this method defers actual rendering via
        :meth:`tk.Tk.after`.

        Inputs
        ------
        frame: numpy.ndarray
            BGR image as produced by OpenCV.

        Outputs
        -------
        None
            The frame is forwarded to :meth:`_update_preview` asynchronously.
        """
        self.master.after(0, lambda f=frame: self._update_preview(f))

    def _update_preview(self, frame: np.ndarray) -> None:
        """Render a scaled preview image in a separate OpenCV window.

        Purpose
        -------
        Avoid issues with embedding previews directly in Tkinter by delegating
        the display to an OpenCV-managed window.  This keeps the GUI responsive
        and works reliably across platforms that struggle with Tk image
        updates.

        Inputs
        ------
        frame: numpy.ndarray
            Full-resolution frame from the pipeline.

        Outputs
        -------
        None
            The ``preview`` window is updated with the latest frame.
        """
        # Lazily create the window to avoid opening it during app start when no
        # frames have been processed yet.
        if self._preview_window is None:
            self._preview_window = "preview"
            # ``WINDOW_NORMAL`` allows user-resizable window for convenience.
            cv2.namedWindow(self._preview_window, cv2.WINDOW_NORMAL)

        # Downscale the frame so high-resolution videos don't overwhelm the
        # display and to keep per-frame processing lightweight.
        h, w = frame.shape[:2]
        scaled = cv2.resize(frame, (int(w * self.preview_scaling), int(h * self.preview_scaling)))
        cv2.imshow(self._preview_window, scaled)
        # ``waitKey`` with small timeout lets OpenCV process its event queue
        # without noticeably blocking the Tkinter loop.
        cv2.waitKey(1)

    def _close_preview_window(self) -> None:
        """Destroy the OpenCV preview window if it was created.

        Purpose
        -------
        Ensure resources tied to the preview are released after pipeline
        completion, allowing subsequent runs to create a fresh window without
        inheriting stale state.

        Inputs
        ------
        None
            Operates on the instance's internal window tracker.

        Outputs
        -------
        None
            Closes the window when present and resets the tracker.
        """
        if self._preview_window is not None:
            cv2.destroyWindow(self._preview_window)
            self._preview_window = None

    def refresh_status(self) -> None:
        """Update status and frame-progress labels with pipeline progress.

        Purpose
        -------
        Poll the :mod:`annotation_pipeline` module for its current execution
        state so users can monitor long-running operations.

        Inputs
        ------
        None
            The method reads global pipeline status.

        Outputs
        -------
        None
            ``self.status`` and ``self.frame_progress`` are updated and the
            method schedules itself to run again.
        """

        st = ap.status
        # Always show the last function name; fall back to "Idle" when nothing
        # has run yet for clarity at application start.
        self.status.set(st.last_function or "Idle")
        if st.total_frames:
            # Expose deterministic frame counts so users know precisely how far
            # along the pipeline is without relying on heuristics.
            self.frame_progress.set(
                f"Frame {st.current_frame}/{st.total_frames}"
            )
        else:
            self.frame_progress.set("Frame: 0/0")
        # Using ``after`` avoids blocking the main loop while providing periodic
        # updates that are sufficient for human perception.
        self.master.after(200, self.refresh_status)


if __name__ == "__main__":
    root = tk.Tk()
    gui = AnnotationGUI(root)
    root.mainloop()
