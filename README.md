# ZED Stereo / Depth Capture GUI

依照 Stereolabs ZED Python API 的 camera、depth、recording 與 SVO playback 流程整理的桌面 GUI。

## 主要功能

- Live ZED camera / SVO playback
- 同時顯示 LEFT、RIGHT、DEPTH VIEW
- GUI 可設定：
  - Output folder
  - SVO / dataset name
  - Resolution
  - FPS
  - Depth mode
  - SVO compression
- Live camera 錄製 `.svo2`
- 匯出目前 frame
- 連續匯出目前來源的 frames
- 整個 SVO 批次匯出
- Depth:
  - 對應 RGB
  - SDK depth visualization
  - float32 `.npy`
  - uint16 millimeter PNG
- Stereo:
  - Left / Right rectified RGB
  - Left / Right disparity `.npy`
  - Right RGB -> Left warp
  - Left RGB -> Right warp
  - Right disparity -> Left coordinates
  - Left disparity -> Right coordinates
  - 雙向 valid mask

## 資料結構

假設：

```text
Output folder = /data/output
SVO / dataset name = rebar_001
```

輸出：

```text
/data/output/
└── rebar_001/
    ├── rebar_001.svo2
    ├── metadata.json
    ├── depth/
    │   ├── rgb/
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── view/
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── npy/
    │   │   ├── 000001.npy
    │   │   └── ...
    │   └── u16_mm/
    │       ├── 000001.png
    │       └── ...
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
```

## Depth

### `depth/rgb/`

Depth 對應的 RGB。

ZED `MEASURE.DEPTH` 對齊 rectified left image，因此：

```text
depth/rgb/000001.png
```

與：

```text
stereo/left_rgb/000001.png
```

是同一張 frame 的左眼 rectified RGB。

程式刻意各保存一份，讓 `depth/` 可以單獨作為 RGB-D dataset 使用。

### `depth/view/`

ZED SDK 的 depth visualization。

只用來顯示與快速檢查，不應把 color/pixel value 當成實際 depth。

### `depth/npy/`

```text
dtype = float32
unit = millimeter
```

來源：

```python
sl.MEASURE.DEPTH
```

例如：

```python
depth = np.load("depth/npy/000001.npy")
print(depth[500, 600])
```

輸出：

```text
1245.37
```

代表 `1245.37 mm`。

### `depth/u16_mm/`

16-bit grayscale PNG。

```text
pixel value = rounded depth in millimeters
0 = invalid
```

讀取：

```python
depth = cv2.imread(
    depth_path,
    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
)
```

或：

```python
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
```

預期：

```text
dtype = uint16
shape = H x W
```

## Stereo

### `stereo/left_rgb/`

Rectified left RGB。

來源：

```python
sl.VIEW.LEFT
```

### `stereo/right_rgb/`

Rectified right RGB。

來源：

```python
sl.VIEW.RIGHT
```

### `stereo/disparity_left_npy/`

Left-reference disparity。

來源：

```python
sl.MEASURE.DISPARITY
```

### `stereo/disparity_right_npy/`

Right-reference disparity。

來源：

```python
sl.MEASURE.DISPARITY_RIGHT
```

程式設定：

```python
init.enable_right_side_measure = True
```

## Warp output

### `right_to_left_rgb_warped/`

將 Right RGB sample 到 Left coordinate system。

Reference：

```text
LEFT
```

Correspondence map：

```text
left disparity
```

### `left_to_right_rgb_warped/`

將 Left RGB sample 到 Right coordinate system。

Reference：

```text
RIGHT
```

Correspondence map：

```text
right disparity
```

### `disparity_right_to_left_warped_npy/`

將 right-reference disparity sample 到 left coordinates。

可直接與：

```text
disparity_left_npy
```

在相同 pixel coordinates 下做 LR consistency analysis。

Invalid pixel 為：

```python
np.nan
```

### `disparity_left_to_right_warped_npy/`

將 left-reference disparity sample 到 right coordinates。

可直接與：

```text
disparity_right_npy
```

在相同 pixel coordinates 下比較。

### `valid_mask_right_to_left/`

```text
255 = valid
0   = invalid
```

對應：

```text
right -> left warp
```

### `valid_mask_left_to_right/`

```text
255 = valid
0   = invalid
```

對應：

```text
left -> right warp
```

## Warp sign

Stereolabs Python enum 文件描述 `DISPARITY` / `DISPARITY_RIGHT` 的 reference sensor 與資料格式，但沒有在該 enum 說明中指定 horizontal warp 的 sign formula。

因此程式不硬編碼 sign。

第一個可用 frame 會比較：

```text
x_source = x_reference + disparity
```

以及：

```text
x_source = x_reference - disparity
```

以 grayscale median photometric MAE 較低者作為該方向的 sign。

選定後會 cache 並寫入：

```text
metadata.json
```

欄位：

```text
stereo.warp.sign_right_to_left
stereo.warp.sign_left_to_right
```

後續同一 source 的 frames 使用相同 sign。

## SVO 儲存

GUI 中設定：

```text
Output folder = /data/output
SVO / dataset name = rebar_001
```

按：

```text
Start SVO Recording
```

產生：

```text
/data/output/rebar_001/rebar_001.svo2
```

因此 SVO 與後續匯出的 depth/stereo 會使用同一 dataset root。

## 建議 workflow

現場：

```text
Open Live Camera
        ↓
設定 Output folder
        ↓
設定 SVO / dataset name
        ↓
Start SVO Recording
        ↓
Stop SVO Recording
```

例如：

```text
output/rebar_001/rebar_001.svo2
```

回實驗室：

```text
Open SVO
        ↓
設定 Output folder
        ↓
Export Loaded SVO All Frames
```

產生：

```text
output/rebar_001/depth/
output/rebar_001/stereo/
```

## 安裝

先安裝 ZED SDK 與 ZED Python API。

再安裝：

```bash
python -m pip install -r requirements.txt
```

Ubuntu 若缺少 Tkinter：

```bash
sudo apt install python3-tk
```

## 執行

```bash
python zed_capture_gui.py
```

## 注意

Live 模式同時連續輸出 PNG、Depth NPY、雙 disparity 與雙向 warp 會造成大量 disk I/O 和 CPU processing。

正式資料採集建議：

```text
現場只錄 SVO2
→
離線 Export Loaded SVO All Frames
```

這樣比較不容易影響相機 capture。
