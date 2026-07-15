#!/usr/bin/env python3
"""
ZED stereo/depth capture GUI.

Output layout
-------------
<output_root>/<svo_stem>/
├── <svo_stem>.svo2                  # live recording only
├── metadata.json
├── depth/
│   ├── rgb/                          # LEFT rectified RGB; aligned with MEASURE.DEPTH
│   ├── view/                         # SDK depth visualization only
│   ├── npy/                          # float32 depth, millimeter
│   └── u16_mm/                       # uint16 PNG, pixel value = rounded depth mm
└── stereo/
    ├── left_rgb/
    ├── right_rgb/
    ├── disparity_left_npy/
    ├── disparity_right_npy/
    ├── right_to_left_rgb_warped/
    ├── left_to_right_rgb_warped/
    ├── disparity_right_to_left_warped_npy/
    ├── disparity_left_to_right_warped_npy/
    ├── valid_mask_right_to_left/
    └── valid_mask_left_to_right/

Warp convention
---------------
ZED exposes left- and right-referenced disparity maps, but the Python enum
documentation only specifies the reference sensor and float map type. To avoid
hard-coding an undocumented disparity sign convention, this application tests
the two horizontal sampling formulas x_source = x_ref + d and x_source = x_ref - d
on the first usable frame for each direction. It selects the sign with lower
grayscale photometric MAE, caches it for the source, and records the selected
sign in metadata.json.

Warped disparity is the OTHER view's disparity sampled into the reference view:
- right disparity -> left coordinates, using the left-reference correspondence map
- left disparity  -> right coordinates, using the right-reference correspondence map
Invalid/out-of-image pixels are NaN in warped disparity and 0 in valid-mask PNG.
"""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import pyzed.sl as sl
except ImportError as exc:
    raise SystemExit(
        "Cannot import pyzed.sl.\n"
        "Install the ZED SDK and ZED Python API first.\n"
        "See: https://docs.stereolabs.com/docs/development/api-languages/python"
    ) from exc


APP_TITLE = "ZED Stereo / Depth Capture"
DISPLAY_SIZE = (560, 315)
GUI_INTERVAL_MS = 15


def safe_stem(text: str) -> str:
    value = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    return value.strip("_") or "zed_capture"


def now_stem() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def mat_copy(mat: sl.Mat) -> np.ndarray:
    """Copy ZED Mat data because SDK buffers are reused by later frames."""
    return np.asarray(mat.get_data()).copy()


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert ZED/OpenCV BGR(A) data to RGB for Pillow/Tkinter."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_standard_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def depth_mm_to_u16(depth_mm: np.ndarray) -> np.ndarray:
    """
    Store depth values as 16-bit grayscale intensities in millimeters.

    0 = invalid.
    Values above uint16 range are clipped.
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    output = np.zeros(depth.shape, dtype=np.uint16)
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        clipped = np.clip(np.rint(depth[valid]), 1, np.iinfo(np.uint16).max)
        output[valid] = clipped.astype(np.uint16)
    return output


def build_dataset_root(output_root: Path, svo_stem: str) -> Path:
    return Path(output_root).expanduser().resolve() / safe_stem(svo_stem)


def build_frame_paths(dataset_root: Path, stem: str) -> dict[str, Path]:
    depth = dataset_root / "depth"
    stereo = dataset_root / "stereo"
    return {
        "depth_rgb": depth / "rgb" / f"{stem}.png",
        "depth_view": depth / "view" / f"{stem}.png",
        "depth_npy": depth / "npy" / f"{stem}.npy",
        "depth_u16": depth / "u16_mm" / f"{stem}.png",
        "left_rgb": stereo / "left_rgb" / f"{stem}.png",
        "right_rgb": stereo / "right_rgb" / f"{stem}.png",
        "disp_left": stereo / "disparity_left_npy" / f"{stem}.npy",
        "disp_right": stereo / "disparity_right_npy" / f"{stem}.npy",
        "warp_r2l_rgb": stereo / "right_to_left_rgb_warped" / f"{stem}.png",
        "warp_l2r_rgb": stereo / "left_to_right_rgb_warped" / f"{stem}.png",
        "warp_r2l_disp": (
            stereo / "disparity_right_to_left_warped_npy" / f"{stem}.npy"
        ),
        "warp_l2r_disp": (
            stereo / "disparity_left_to_right_warped_npy" / f"{stem}.npy"
        ),
        "mask_r2l": stereo / "valid_mask_right_to_left" / f"{stem}.png",
        "mask_l2r": stereo / "valid_mask_left_to_right" / f"{stem}.png",
    }


def make_horizontal_map(
    disparity_ref: np.ndarray,
    sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a horizontal cv2.remap correspondence map.

    x_source = x_reference + sign * disparity_reference

    Returns map_x, map_y, valid_mask.
    """
    disparity = np.asarray(disparity_ref, dtype=np.float32)
    if disparity.ndim != 2:
        raise ValueError(f"Disparity must be HxW, got {disparity.shape}")
    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or +1, got {sign}")

    height, width = disparity.shape
    grid_x = np.arange(width, dtype=np.float32)[None, :]
    grid_y = np.arange(height, dtype=np.float32)[:, None]

    map_x = np.broadcast_to(grid_x, (height, width)).copy()
    map_y = np.broadcast_to(grid_y, (height, width)).copy()
    map_x += float(sign) * disparity

    valid = (
        np.isfinite(disparity)
        & np.isfinite(map_x)
        & (map_x >= 0.0)
        & (map_x <= float(width - 1))
    )

    # OpenCV remap receives finite coordinates even for invalid disparity pixels.
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid


def remap_array(
    source: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid: np.ndarray,
    *,
    interpolation: int,
    invalid_value: float | int,
) -> np.ndarray:
    """Remap image/measure and explicitly overwrite invalid pixels."""
    warped = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    warped = warped.copy()
    if warped.ndim == 2:
        warped[~valid] = invalid_value
    else:
        warped[~valid, ...] = invalid_value
    return warped


def photometric_mae(
    reference: np.ndarray,
    warped: np.ndarray,
    valid: np.ndarray,
) -> float:
    """Robust grayscale MAE used only to infer disparity sign."""
    if valid.shape != reference.shape[:2]:
        return float("inf")

    # Remove a small boundary around invalid regions to reduce interpolation-edge bias.
    valid_u8 = valid.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    score_mask = cv2.erode(valid_u8, kernel, iterations=1) > 0

    count = int(np.count_nonzero(score_mask))
    minimum = max(500, int(valid.size * 0.01))
    if count < minimum:
        return float("inf")

    ref_gray = image_to_gray(reference).astype(np.float32)
    warped_gray = image_to_gray(warped).astype(np.float32)
    error = np.abs(ref_gray - warped_gray)
    return float(np.median(error[score_mask]))


def infer_warp_sign(
    source: np.ndarray,
    reference: np.ndarray,
    disparity_ref: np.ndarray,
) -> int:
    """
    Infer whether ZED disparity should be sampled as x+d or x-d.

    The lower median grayscale photometric error wins.
    """
    candidates: dict[int, float] = {}
    for sign in (1, -1):
        map_x, map_y, valid = make_horizontal_map(disparity_ref, sign)
        warped = remap_array(
            source,
            map_x,
            map_y,
            valid,
            interpolation=cv2.INTER_LINEAR,
            invalid_value=0,
        )
        candidates[sign] = photometric_mae(reference, warped, valid)

    best_sign = min(candidates, key=candidates.get)
    if not np.isfinite(candidates[best_sign]):
        # Conservative fallback. The sign is still written to metadata.
        return 1
    return int(best_sign)


def make_warp_bundle(
    *,
    source_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    disparity_ref: np.ndarray,
    disparity_source: np.ndarray,
    sign: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Warp source RGB and source-view disparity into reference coordinates.

    The correspondence map is defined by the reference-view disparity.
    """
    if sign is None:
        sign = infer_warp_sign(source_rgb, reference_rgb, disparity_ref)

    map_x, map_y, valid = make_horizontal_map(disparity_ref, sign)

    warped_rgb = remap_array(
        source_rgb,
        map_x,
        map_y,
        valid,
        interpolation=cv2.INTER_LINEAR,
        invalid_value=0,
    )
    warped_disparity = remap_array(
        np.asarray(disparity_source, dtype=np.float32),
        map_x,
        map_y,
        valid,
        interpolation=cv2.INTER_LINEAR,
        invalid_value=np.nan,
    ).astype(np.float32, copy=False)

    valid &= np.isfinite(warped_disparity)
    warped_disparity[~valid] = np.nan
    valid_mask = valid.astype(np.uint8) * 255
    return warped_rgb, warped_disparity, valid_mask, sign


def save_frame_bundle(
    dataset_root: Path,
    stem: str,
    *,
    left: np.ndarray,
    right: np.ndarray,
    depth_view: np.ndarray,
    depth_mm: Optional[np.ndarray],
    disparity_left: Optional[np.ndarray],
    disparity_right: Optional[np.ndarray],
    warp_sign_r2l: Optional[int],
    warp_sign_l2r: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    """
    Save one synchronized frame using the depth/ and stereo/ layout.

    depth/rgb intentionally duplicates the left rectified image because ZED
    MEASURE.DEPTH is aligned with the left image.
    """
    paths = build_frame_paths(dataset_root, stem)

    # RGB + visualization
    save_standard_png(paths["depth_rgb"], left)
    save_standard_png(paths["depth_view"], depth_view)
    save_standard_png(paths["left_rgb"], left)
    save_standard_png(paths["right_rgb"], right)

    # Depth
    if depth_mm is not None:
        paths["depth_npy"].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths["depth_npy"], np.asarray(depth_mm, dtype=np.float32))
        save_standard_png(paths["depth_u16"], depth_mm_to_u16(depth_mm))

    # Raw disparity + bidirectional warp
    if disparity_left is not None and disparity_right is not None:
        paths["disp_left"].parent.mkdir(parents=True, exist_ok=True)
        paths["disp_right"].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths["disp_left"], np.asarray(disparity_left, dtype=np.float32))
        np.save(paths["disp_right"], np.asarray(disparity_right, dtype=np.float32))

        warp_r2l_rgb, warp_r2l_disp, mask_r2l, warp_sign_r2l = make_warp_bundle(
            source_rgb=right,
            reference_rgb=left,
            disparity_ref=disparity_left,
            disparity_source=disparity_right,
            sign=warp_sign_r2l,
        )
        warp_l2r_rgb, warp_l2r_disp, mask_l2r, warp_sign_l2r = make_warp_bundle(
            source_rgb=left,
            reference_rgb=right,
            disparity_ref=disparity_right,
            disparity_source=disparity_left,
            sign=warp_sign_l2r,
        )

        save_standard_png(paths["warp_r2l_rgb"], warp_r2l_rgb)
        save_standard_png(paths["warp_l2r_rgb"], warp_l2r_rgb)

        paths["warp_r2l_disp"].parent.mkdir(parents=True, exist_ok=True)
        paths["warp_l2r_disp"].parent.mkdir(parents=True, exist_ok=True)
        np.save(paths["warp_r2l_disp"], warp_r2l_disp)
        np.save(paths["warp_l2r_disp"], warp_l2r_disp)

        save_standard_png(paths["mask_r2l"], mask_r2l)
        save_standard_png(paths["mask_l2r"], mask_l2r)

    return warp_sign_r2l, warp_sign_l2r


def write_dataset_metadata(
    dataset_root: Path,
    *,
    source_mode: str,
    source_svo: Optional[Path],
    svo_record_path: Optional[Path],
    depth_mode: str,
    warp_sign_r2l: Optional[int],
    warp_sign_l2r: Optional[int],
    extra: Optional[dict] = None,
) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_mode": source_mode,
        "source_svo": str(source_svo) if source_svo is not None else None,
        "svo_record_path": (
            str(svo_record_path) if svo_record_path is not None else None
        ),
        "depth": {
            "reference_rgb": "depth/rgb; identical to rectified LEFT RGB",
            "aligned_to": "left image",
            "npy_dtype": "float32",
            "unit": "millimeter",
            "u16_png": (
                "uint16 grayscale; pixel value = rounded depth in millimeters; "
                "0 = invalid; values above 65535 mm clipped"
            ),
            "view": "SDK depth visualization only; not measurement data",
            "depth_mode": depth_mode,
        },
        "stereo": {
            "left_right_images": "ZED SDK rectified VIEW.LEFT / VIEW.RIGHT",
            "disparity_dtype": "float32",
            "right_side_measure_enabled": True,
            "warp": {
                "right_to_left": (
                    "right RGB and right disparity sampled into left coordinates "
                    "using the left-reference disparity correspondence map"
                ),
                "left_to_right": (
                    "left RGB and left disparity sampled into right coordinates "
                    "using the right-reference disparity correspondence map"
                ),
                "formula": "x_source = x_reference + sign * disparity_reference",
                "sign_selection": (
                    "sign is inferred on first usable frame per direction by "
                    "minimum median grayscale photometric MAE, then cached"
                ),
                "sign_right_to_left": warp_sign_r2l,
                "sign_left_to_right": warp_sign_l2r,
                "warped_disparity_invalid_value": "NaN",
                "valid_mask": "uint8 PNG; 255 = valid, 0 = invalid",
            },
        },
    }
    if extra:
        metadata.update(extra)

    (dataset_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class ZEDCaptureGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1800x790")
        self.minsize(1250, 700)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cam = sl.Camera()
        self.runtime = sl.RuntimeParameters()

        self.left_mat = sl.Mat()
        self.right_mat = sl.Mat()
        self.depth_view_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.disparity_left_mat = sl.Mat()
        self.disparity_right_mat = sl.Mat()

        self.source_mode: Optional[str] = None
        self.svo_path: Optional[Path] = None
        self.svo_paused = False
        self._updating_seek = False

        self.is_svo_recording = False
        self.svo_record_path: Optional[Path] = None

        self.is_sequence_exporting = False
        self.sequence_frame_index = 0

        self.last_left: Optional[np.ndarray] = None
        self.last_right: Optional[np.ndarray] = None
        self.last_depth_view: Optional[np.ndarray] = None

        self.warp_sign_r2l: Optional[int] = None
        self.warp_sign_l2r: Optional[int] = None

        self.export_thread: Optional[threading.Thread] = None
        self.worker_messages: queue.Queue[tuple] = queue.Queue()
        self.closing = False

        self._build_ui()
        self.after(GUI_INTERVAL_MS, self._update_loop)

    # ----------------------------- UI ---------------------------------

    def _build_ui(self) -> None:
        source_frame = ttk.LabelFrame(self, text="Source / Camera", padding=8)
        source_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Button(
            source_frame, text="Open Live Camera", command=self.open_live
        ).grid(row=0, column=0, padx=4, pady=3)
        ttk.Button(source_frame, text="Open SVO", command=self.open_svo).grid(
            row=0, column=1, padx=4, pady=3
        )

        ttk.Label(source_frame, text="Resolution").grid(
            row=0, column=2, padx=(14, 3)
        )
        self.resolution_var = tk.StringVar(value="HD720")
        ttk.Combobox(
            source_frame,
            textvariable=self.resolution_var,
            values=("HD2K", "HD1080", "HD720", "VGA"),
            width=9,
            state="readonly",
        ).grid(row=0, column=3, padx=3)

        ttk.Label(source_frame, text="FPS").grid(
            row=0, column=4, padx=(10, 3)
        )
        self.fps_var = tk.StringVar(value="30")
        ttk.Combobox(
            source_frame,
            textvariable=self.fps_var,
            values=("15", "30", "60", "100"),
            width=5,
            state="readonly",
        ).grid(row=0, column=5, padx=3)

        ttk.Label(source_frame, text="Depth mode").grid(
            row=0, column=6, padx=(10, 3)
        )
        self.depth_mode_var = tk.StringVar(value="NEURAL")
        ttk.Combobox(
            source_frame,
            textvariable=self.depth_mode_var,
            values=("NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY", "PERFORMANCE"),
            width=13,
            state="readonly",
        ).grid(row=0, column=7, padx=3)

        self.pause_button = ttk.Button(
            source_frame,
            text="Pause SVO",
            command=self.toggle_svo_pause,
            state=tk.DISABLED,
        )
        self.pause_button.grid(row=0, column=8, padx=(14, 4), pady=3)

        output_frame = ttk.LabelFrame(self, text="Output / SVO", padding=8)
        output_frame.pack(fill=tk.X, padx=8, pady=4)
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output folder").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        self.output_root_var = tk.StringVar(
            value=str((Path.cwd() / "output").resolve())
        )
        ttk.Entry(output_frame, textvariable=self.output_root_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=3
        )
        ttk.Button(
            output_frame, text="Browse...", command=self.choose_output_root
        ).grid(row=0, column=2, padx=4, pady=3)

        ttk.Label(output_frame, text="SVO / dataset name").grid(
            row=1, column=0, sticky="w", padx=4, pady=3
        )
        self.svo_name_var = tk.StringVar(value=f"zed_{now_stem()}")
        ttk.Entry(output_frame, textvariable=self.svo_name_var).grid(
            row=1, column=1, sticky="ew", padx=4, pady=3
        )

        ttk.Label(output_frame, text="Compression").grid(
            row=1, column=2, padx=(10, 3)
        )
        self.compression_var = tk.StringVar(value="H265")
        ttk.Combobox(
            output_frame,
            textvariable=self.compression_var,
            values=("H265", "H264", "LOSSLESS"),
            width=10,
            state="readonly",
        ).grid(row=1, column=3, padx=3)

        self.svo_record_button = ttk.Button(
            output_frame,
            text="Start SVO Recording",
            command=self.toggle_svo_recording,
        )
        self.svo_record_button.grid(row=0, column=3, padx=8, pady=3)

        action_frame = ttk.LabelFrame(self, text="Frame Export", padding=8)
        action_frame.pack(fill=tk.X, padx=8, pady=4)

        self.export_depth_var = tk.BooleanVar(value=True)
        self.export_stereo_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            action_frame,
            text="Depth (.npy + u16 PNG + aligned RGB + depth view)",
            variable=self.export_depth_var,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4)

        ttk.Checkbutton(
            action_frame,
            text="Stereo (L/R RGB + L/R disparity + warped RGB/disparity/masks)",
            variable=self.export_stereo_var,
        ).grid(row=0, column=3, columnspan=4, sticky="w", padx=4)

        self.sequence_button = ttk.Button(
            action_frame,
            text="Start Frame Export Sequence",
            command=self.toggle_sequence_export,
        )
        self.sequence_button.grid(row=1, column=0, padx=4, pady=4)

        ttk.Button(
            action_frame,
            text="Export Current Frame",
            command=self.export_current_frame,
        ).grid(row=1, column=1, padx=4, pady=4)

        self.export_svo_button = ttk.Button(
            action_frame,
            text="Export Loaded SVO All Frames",
            command=self.export_loaded_svo,
        )
        self.export_svo_button.grid(row=1, column=2, padx=4, pady=4)

        ttk.Label(
            action_frame,
            text="Dataset root: <Output folder>/<SVO name>/",
        ).grid(row=1, column=3, columnspan=4, sticky="w", padx=12)

        image_area = ttk.Frame(self, padding=(8, 4))
        image_area.pack(fill=tk.BOTH, expand=True)

        self.left_label = self._create_view_panel(image_area, "LEFT", 0)
        self.right_label = self._create_view_panel(image_area, "RIGHT", 1)
        self.depth_label = self._create_view_panel(image_area, "DEPTH VIEW", 2)

        svo_controls = ttk.Frame(self, padding=(10, 4))
        svo_controls.pack(fill=tk.X)

        ttk.Label(svo_controls, text="SVO frame").pack(side=tk.LEFT)
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_scale = ttk.Scale(
            svo_controls,
            variable=self.seek_var,
            from_=0,
            to=1,
            command=self._on_seek,
            state=tk.DISABLED,
        )
        self.seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.frame_position_label = ttk.Label(
            svo_controls, text="- / -", width=18
        )
        self.frame_position_label.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(
            value="Ready. Open a live ZED camera or an SVO file."
        )
        ttk.Label(
            self,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor="w",
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _create_view_panel(
        self,
        parent: ttk.Frame,
        title: str,
        column: int,
    ) -> ttk.Label:
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=0, column=column, padx=5, sticky="nsew")
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(0, weight=1)

        label = ttk.Label(frame, text="No frame", anchor="center")
        label.pack(fill=tk.BOTH, expand=True)
        return label

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def choose_output_root(self) -> None:
        directory = filedialog.askdirectory(
            title="Select output root",
            initialdir=self.output_root_var.get(),
        )
        if directory:
            self.output_root_var.set(directory)

    def _dataset_name(self) -> str:
        return safe_stem(self.svo_name_var.get())

    def _dataset_root(self) -> Path:
        return build_dataset_root(
            Path(self.output_root_var.get()),
            self._dataset_name(),
        )

    # ------------------------ Camera lifecycle -------------------------

    def _make_init_parameters(
        self,
        svo_path: Optional[Path] = None,
    ) -> sl.InitParameters:
        init = sl.InitParameters()
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode_var.get())
        init.enable_right_side_measure = True

        if svo_path is not None:
            init.set_from_svo_file(str(svo_path))
            init.svo_real_time_mode = False
        else:
            init.camera_resolution = getattr(
                sl.RESOLUTION, self.resolution_var.get()
            )
            init.camera_fps = int(self.fps_var.get())

        return init

    def _reset_warp_signs(self) -> None:
        self.warp_sign_r2l = None
        self.warp_sign_l2r = None

    def _close_source(self) -> None:
        if self.is_svo_recording:
            try:
                self.cam.disable_recording()
            except Exception:
                pass
            self.is_svo_recording = False
            self.svo_record_button.config(text="Start SVO Recording")

        self.is_sequence_exporting = False
        self.sequence_button.config(text="Start Frame Export Sequence")

        if self.source_mode is not None:
            try:
                self.cam.close()
            except Exception:
                pass

        self.cam = sl.Camera()
        self.source_mode = None
        self.svo_path = None
        self.svo_paused = False
        self.pause_button.config(state=tk.DISABLED, text="Pause SVO")
        self.seek_scale.config(state=tk.DISABLED, from_=0, to=1)
        self.frame_position_label.config(text="- / -")
        self.last_left = None
        self.last_right = None
        self.last_depth_view = None
        self._reset_warp_signs()

    def open_live(self) -> None:
        self._close_source()
        init = self._make_init_parameters()

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.cam.close()
            messagebox.showerror(
                "ZED open error",
                f"Failed to open live camera:\n{status}",
            )
            self.set_status(f"Live camera open failed: {status}")
            return

        self.source_mode = "live"
        self.sequence_frame_index = 0
        self.svo_name_var.set(f"zed_{now_stem()}")

        config = self.cam.get_camera_information().camera_configuration
        self.set_status(
            f"LIVE | {config.resolution.width}x{config.resolution.height} "
            f"@ {config.fps} FPS | depth={self.depth_mode_var.get()}"
        )

    def open_svo(self) -> None:
        path = filedialog.askopenfilename(
            title="Open ZED SVO",
            filetypes=[
                ("ZED SVO", "*.svo *.svo2"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        svo_path = Path(path)
        self._close_source()
        init = self._make_init_parameters(svo_path)

        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            self.cam.close()
            messagebox.showerror(
                "SVO open error",
                f"Failed to open SVO:\n{status}",
            )
            self.set_status(f"SVO open failed: {status}")
            return

        self.source_mode = "svo"
        self.svo_path = svo_path
        self.svo_paused = False
        self.svo_name_var.set(svo_path.stem)

        total = max(int(self.cam.get_svo_number_of_frames()), 1)
        self.seek_scale.config(
            state=tk.NORMAL,
            from_=0,
            to=max(total - 1, 0),
        )
        self.pause_button.config(state=tk.NORMAL, text="Pause SVO")
        self.set_status(
            f"SVO PLAYBACK | {svo_path} | output={self._dataset_root()}"
        )

    # --------------------------- Frame loop ----------------------------

    def _update_loop(self) -> None:
        if self.closing:
            return

        self._drain_worker_messages()

        try:
            if self.source_mode is not None:
                if self.source_mode != "svo" or not self.svo_paused:
                    error = self.cam.grab(self.runtime)

                    if error == sl.ERROR_CODE.SUCCESS:
                        self._retrieve_and_display_frame()
                    elif (
                        self.source_mode == "svo"
                        and error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED
                    ):
                        self.svo_paused = True
                        self.pause_button.config(text="Play SVO")
                        self.set_status("SVO end reached.")
                    else:
                        self.set_status(f"Grab error: {error}")
        except Exception as exc:
            self.set_status(f"Runtime error: {exc}")

        self.after(GUI_INTERVAL_MS, self._update_loop)

    def _drain_worker_messages(self) -> None:
        while True:
            try:
                message = self.worker_messages.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "status":
                self.set_status(message[1])
            elif kind == "finish":
                _, success, text = message
                self._finish_svo_export(success, text)

    def _retrieve_and_display_frame(self) -> None:
        self.cam.retrieve_image(self.left_mat, sl.VIEW.LEFT)
        self.cam.retrieve_image(self.right_mat, sl.VIEW.RIGHT)
        self.cam.retrieve_image(self.depth_view_mat, sl.VIEW.DEPTH)

        self.last_left = mat_copy(self.left_mat)
        self.last_right = mat_copy(self.right_mat)
        self.last_depth_view = mat_copy(self.depth_view_mat)

        self._show_image(self.left_label, self.last_left)
        self._show_image(self.right_label, self.last_right)
        self._show_image(self.depth_label, self.last_depth_view)

        if self.source_mode == "svo":
            current = int(self.cam.get_svo_position())
            total = int(self.cam.get_svo_number_of_frames())
            self._updating_seek = True
            self.seek_var.set(current)
            self._updating_seek = False
            self.frame_position_label.config(
                text=f"{current} / {max(total - 1, 0)}"
            )

        if self.is_sequence_exporting:
            self._export_current_frame_to_dataset(sequence_mode=True)

    def _show_image(self, label: ttk.Label, image: np.ndarray) -> None:
        rgb = image_to_rgb(image)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail(DISPLAY_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo, text="")
        label.image = photo

    # --------------------------- SVO control ---------------------------

    def toggle_svo_pause(self) -> None:
        if self.source_mode != "svo":
            return
        self.svo_paused = not self.svo_paused
        self.pause_button.config(
            text="Play SVO" if self.svo_paused else "Pause SVO"
        )
        self.set_status(
            "SVO paused."
            if self.svo_paused
            else "SVO playback resumed."
        )

    def _on_seek(self, value: str) -> None:
        if self._updating_seek or self.source_mode != "svo":
            return

        self.cam.set_svo_position(int(float(value)))
        self._reset_warp_signs()

        if self.svo_paused:
            error = self.cam.grab(self.runtime)
            if error == sl.ERROR_CODE.SUCCESS:
                self._retrieve_and_display_frame()

    # -------------------------- SVO recording --------------------------

    def toggle_svo_recording(self) -> None:
        if self.is_svo_recording:
            self.cam.disable_recording()
            self.is_svo_recording = False
            self.svo_record_button.config(text="Start SVO Recording")
            write_dataset_metadata(
                self._dataset_root(),
                source_mode="live",
                source_svo=None,
                svo_record_path=self.svo_record_path,
                depth_mode=self.depth_mode_var.get(),
                warp_sign_r2l=self.warp_sign_r2l,
                warp_sign_l2r=self.warp_sign_l2r,
            )
            self.set_status(f"SVO recording stopped: {self.svo_record_path}")
            return

        if self.source_mode != "live":
            messagebox.showwarning(
                "Live camera required",
                "SVO recording is enabled only for a live camera source.",
            )
            return

        dataset_root = self._dataset_root()
        dataset_root.mkdir(parents=True, exist_ok=True)

        svo_name = self._dataset_name()
        path = dataset_root / f"{svo_name}.svo2"

        if path.exists():
            overwrite = messagebox.askyesno(
                "SVO exists",
                f"{path}\nalready exists. Overwrite it?",
            )
            if not overwrite:
                return
            path.unlink()

        compression = getattr(
            sl.SVO_COMPRESSION_MODE,
            self.compression_var.get(),
        )
        params = sl.RecordingParameters(str(path), compression)
        error = self.cam.enable_recording(params)

        if error != sl.ERROR_CODE.SUCCESS:
            messagebox.showerror(
                "Recording error",
                f"Cannot start SVO recording:\n{error}",
            )
            self.set_status(f"SVO recording failed: {error}")
            return

        self.is_svo_recording = True
        self.svo_record_path = path
        self.svo_record_button.config(text="Stop SVO Recording")

        write_dataset_metadata(
            dataset_root,
            source_mode="live",
            source_svo=None,
            svo_record_path=path,
            depth_mode=self.depth_mode_var.get(),
            warp_sign_r2l=self.warp_sign_r2l,
            warp_sign_l2r=self.warp_sign_l2r,
        )
        self.set_status(f"Recording SVO: {path}")

    # ------------------------- Frame exporting -------------------------

    def toggle_sequence_export(self) -> None:
        if self.is_sequence_exporting:
            self.is_sequence_exporting = False
            self.sequence_button.config(text="Start Frame Export Sequence")
            self._write_current_metadata()
            self.set_status(
                f"Frame export sequence stopped: {self._dataset_root()}"
            )
            return

        if self.source_mode is None:
            messagebox.showwarning(
                "No source",
                "Open a live camera or SVO file first.",
            )
            return

        self._dataset_root().mkdir(parents=True, exist_ok=True)
        self.sequence_frame_index = 0
        self.is_sequence_exporting = True
        self.sequence_button.config(text="Stop Frame Export Sequence")
        self.set_status(
            f"Exporting frame sequence to: {self._dataset_root()}"
        )

    def export_current_frame(self) -> None:
        if self.source_mode is None or self.last_left is None:
            messagebox.showwarning(
                "No frame",
                "No ZED frame is currently available.",
            )
            return
        try:
            self._export_current_frame_to_dataset(sequence_mode=False)
            self.set_status(
                f"Current frame exported: {self._dataset_root()}"
            )
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self.set_status(f"Current-frame export failed: {exc}")

    def _retrieve_raw_measures(
        self,
        *,
        need_depth: bool,
        need_stereo: bool,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        depth = None
        disp_left = None
        disp_right = None

        if need_depth:
            self.cam.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
            depth = mat_copy(self.depth_mat).astype(
                np.float32,
                copy=False,
            )

        if need_stereo:
            self.cam.retrieve_measure(
                self.disparity_left_mat,
                sl.MEASURE.DISPARITY,
            )
            self.cam.retrieve_measure(
                self.disparity_right_mat,
                sl.MEASURE.DISPARITY_RIGHT,
            )
            disp_left = mat_copy(self.disparity_left_mat).astype(
                np.float32,
                copy=False,
            )
            disp_right = mat_copy(self.disparity_right_mat).astype(
                np.float32,
                copy=False,
            )

        return depth, disp_left, disp_right

    def _current_frame_stem(self, sequence_mode: bool) -> str:
        if self.source_mode == "svo":
            return f"{int(self.cam.get_svo_position()):06d}"
        if sequence_mode:
            return f"{self.sequence_frame_index:06d}"
        return f"live_{now_stem()}"

    def _export_current_frame_to_dataset(
        self,
        *,
        sequence_mode: bool,
    ) -> None:
        if (
            self.last_left is None
            or self.last_right is None
            or self.last_depth_view is None
        ):
            return

        need_depth = self.export_depth_var.get()
        need_stereo = self.export_stereo_var.get()
        depth, disp_left, disp_right = self._retrieve_raw_measures(
            need_depth=need_depth,
            need_stereo=need_stereo,
        )

        stem = self._current_frame_stem(sequence_mode)
        self.warp_sign_r2l, self.warp_sign_l2r = save_frame_bundle(
            self._dataset_root(),
            stem,
            left=self.last_left,
            right=self.last_right,
            depth_view=self.last_depth_view,
            depth_mm=depth,
            disparity_left=disp_left,
            disparity_right=disp_right,
            warp_sign_r2l=self.warp_sign_r2l,
            warp_sign_l2r=self.warp_sign_l2r,
        )

        if sequence_mode and self.source_mode == "live":
            self.sequence_frame_index += 1

        self._write_current_metadata()

    def _write_current_metadata(self) -> None:
        write_dataset_metadata(
            self._dataset_root(),
            source_mode=self.source_mode or "unknown",
            source_svo=self.svo_path,
            svo_record_path=self.svo_record_path,
            depth_mode=self.depth_mode_var.get(),
            warp_sign_r2l=self.warp_sign_r2l,
            warp_sign_l2r=self.warp_sign_l2r,
        )

    # ------------------------- Full SVO export -------------------------

    def export_loaded_svo(self) -> None:
        if self.source_mode != "svo" or self.svo_path is None:
            messagebox.showwarning(
                "No SVO",
                "Open an SVO file first.",
            )
            return

        if self.export_thread is not None and self.export_thread.is_alive():
            messagebox.showinfo(
                "Export running",
                "An SVO export is already running.",
            )
            return

        svo_path = self.svo_path
        dataset_root = self._dataset_root()
        need_depth = self.export_depth_var.get()
        need_stereo = self.export_stereo_var.get()
        depth_mode_name = self.depth_mode_var.get()

        dataset_root.mkdir(parents=True, exist_ok=True)
        self.export_svo_button.config(state=tk.DISABLED)
        self.set_status(
            f"Starting full SVO export: {svo_path.name} -> {dataset_root}"
        )

        self.export_thread = threading.Thread(
            target=self._export_svo_worker,
            args=(
                svo_path,
                dataset_root,
                need_depth,
                need_stereo,
                depth_mode_name,
            ),
            daemon=True,
        )
        self.export_thread.start()

    def _export_svo_worker(
        self,
        svo_path: Path,
        dataset_root: Path,
        need_depth: bool,
        need_stereo: bool,
        depth_mode_name: str,
    ) -> None:
        zed = sl.Camera()
        warp_sign_r2l: Optional[int] = None
        warp_sign_l2r: Optional[int] = None

        try:
            init = sl.InitParameters()
            init.set_from_svo_file(str(svo_path))
            init.svo_real_time_mode = False
            init.coordinate_units = sl.UNIT.MILLIMETER
            init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode_name)
            init.enable_right_side_measure = True

            error = zed.open(init)
            if error != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f"Failed to open SVO for export: {error}"
                )

            runtime = sl.RuntimeParameters()
            left_mat = sl.Mat()
            right_mat = sl.Mat()
            depth_view_mat = sl.Mat()
            depth_mat = sl.Mat()
            disp_left_mat = sl.Mat()
            disp_right_mat = sl.Mat()

            total = int(zed.get_svo_number_of_frames())
            exported = 0

            while True:
                error = zed.grab(runtime)

                if error == sl.ERROR_CODE.SUCCESS:
                    position = int(zed.get_svo_position())
                    stem = f"{position:06d}"

                    zed.retrieve_image(left_mat, sl.VIEW.LEFT)
                    zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
                    zed.retrieve_image(depth_view_mat, sl.VIEW.DEPTH)

                    left = mat_copy(left_mat)
                    right = mat_copy(right_mat)
                    depth_view = mat_copy(depth_view_mat)

                    depth = None
                    disp_left = None
                    disp_right = None

                    if need_depth:
                        zed.retrieve_measure(
                            depth_mat,
                            sl.MEASURE.DEPTH,
                        )
                        depth = mat_copy(depth_mat).astype(
                            np.float32,
                            copy=False,
                        )

                    if need_stereo:
                        zed.retrieve_measure(
                            disp_left_mat,
                            sl.MEASURE.DISPARITY,
                        )
                        zed.retrieve_measure(
                            disp_right_mat,
                            sl.MEASURE.DISPARITY_RIGHT,
                        )
                        disp_left = mat_copy(disp_left_mat).astype(
                            np.float32,
                            copy=False,
                        )
                        disp_right = mat_copy(disp_right_mat).astype(
                            np.float32,
                            copy=False,
                        )

                    warp_sign_r2l, warp_sign_l2r = save_frame_bundle(
                        dataset_root,
                        stem,
                        left=left,
                        right=right,
                        depth_view=depth_view,
                        depth_mm=depth,
                        disparity_left=disp_left,
                        disparity_right=disp_right,
                        warp_sign_r2l=warp_sign_r2l,
                        warp_sign_l2r=warp_sign_l2r,
                    )

                    exported += 1
                    if exported % 10 == 0 or exported == 1:
                        self.worker_messages.put(
                            (
                                "status",
                                f"Exporting SVO: {exported}/{total} "
                                f"frames -> {dataset_root}",
                            )
                        )
                    continue

                if error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break

                raise RuntimeError(
                    f"SVO grab error during export: {error}"
                )

            write_dataset_metadata(
                dataset_root,
                source_mode="svo",
                source_svo=svo_path,
                svo_record_path=None,
                depth_mode=depth_mode_name,
                warp_sign_r2l=warp_sign_r2l,
                warp_sign_l2r=warp_sign_l2r,
                extra={
                    "total_frames_reported": total,
                    "frames_exported": exported,
                },
            )

            self.worker_messages.put(
                (
                    "finish",
                    True,
                    f"SVO export completed: {exported} frames "
                    f"-> {dataset_root}",
                )
            )
        except Exception as exc:
            self.worker_messages.put(("finish", False, str(exc)))
        finally:
            try:
                zed.close()
            except Exception:
                pass

    def _finish_svo_export(
        self,
        success: bool,
        message: str,
    ) -> None:
        self.export_svo_button.config(state=tk.NORMAL)
        self.set_status(message)
        if success:
            messagebox.showinfo("SVO export", message)
        else:
            messagebox.showerror("SVO export error", message)

    # ----------------------------- Close -------------------------------

    def on_close(self) -> None:
        self.closing = True
        self._close_source()
        self.destroy()


def main() -> None:
    app = ZEDCaptureGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
