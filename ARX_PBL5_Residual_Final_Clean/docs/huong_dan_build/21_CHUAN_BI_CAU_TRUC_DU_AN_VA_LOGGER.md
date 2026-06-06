# Chuẩn bị cấu trúc dự án và logger

## 1. Cấu trúc tối thiểu

Khi build từ đầu, tạo folder như sau:

```text
ARX_PBL5_Residual_Final_Clean/
  src/
    arx_backbone_pipeline.py
    arx_residual_experiment.py
    debug_walkthrough.py
  results/
  docs/
    ly_thuyet/
    huong_dan_build/
    du_lieu/
    bao_ve/
```

`src` chứa code. `results` chứa file sinh ra sau khi chạy. `docs` chứa giải thích.

## 2. Import cần có

Trong `arx_backbone_pipeline.py`, bắt đầu bằng:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
```

Giải thích:

- `argparse`: đọc tham số từ command line;
- `json`: ghi `metrics.json`;
- `dataclass`: gom config thành object dễ đọc;
- `Path`: xử lý đường dẫn;
- `Any`: type hint cho dữ liệu linh hoạt;
- `numpy`: tính toán ma trận;
- `pandas`: xử lý CSV/time-series.

Trong `arx_residual_experiment.py`, cần thêm:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import arx_backbone_pipeline as base
```

`base` nghĩa là residual sẽ dùng lại toàn bộ hàm ARX backbone.

## 3. Khai báo đường dẫn

Viết:

```python
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
```

Ý nghĩa:

- `__file__`: file Python hiện tại;
- `parents[1]`: đi lên một cấp từ `src` về folder project;
- `RESULTS_DIR`: nơi lưu kết quả.

Log nên in:

```python
print(ROOT)
print(RESULTS_DIR)
```

Kỳ vọng:

```text
...\ARX_PBL5_Residual_Final_Clean
...\ARX_PBL5_Residual_Final_Clean\results
```

## 4. Viết `OutdoorConfig`

`OutdoorConfig` là hộp đựng tham số cho dữ liệu và ARX:

```python
@dataclass(frozen=True)
class OutdoorConfig:
    days: int = 16
    sampling_seconds: int = 20
    seed: int = 305031
    train_ratio: float = 0.70
    val_ratio: float = 0.15
```

Cách hiểu cho người mới:

- `days`: số ngày dữ liệu mô phỏng;
- `sampling_seconds`: bao lâu lấy một mẫu;
- `seed`: để sinh dữ liệu lặp lại được;
- `train_ratio`, `val_ratio`: chia dữ liệu.

Log sau khi tạo:

```python
cfg = OutdoorConfig()
print(cfg)
```

Kỳ vọng:

```text
OutdoorConfig(days=16, sampling_seconds=20, ...)
```

## 5. Viết `ArxSpec`

`ArxSpec` mô tả một cấu hình ARX:

```python
@dataclass(frozen=True)
class ArxSpec:
    na: int
    nb: int
    nk: int
    alpha: float
```

Ý nghĩa:

- `na`: số lag của output;
- `nb`: số lag của mỗi input;
- `nk`: độ trễ input;
- `alpha`: Ridge regularization.

Thuộc tính `name`:

```python
@property
def name(self) -> str:
    return f"ARX_na{self.na}_nb{self.nb}_nk{self.nk}_alpha{self.alpha:g}"
```

Log:

```python
spec = ArxSpec(12, 6, 1, 10.0)
print(spec.name)
```

Kỳ vọng:

```text
ARX_na12_nb6_nk1_alpha10
```

## 6. Chọn input chính

Trong `INSIDE_INPUT_COLS`, mỗi tên là một cột model được phép dùng:

```python
INSIDE_INPUT_COLS = (
    "Temperature_In",
    "Humidity_In",
    "Light_In",
    "Drip",
    "Mist",
    "Fan",
    "Light_log",
    "TempIn_x_HumiIn",
    "TempIn_x_Light",
    "HumiIn_x_Light",
    "Indoor_Dryness",
    "VPD_Proxy_In",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
)
```

Không còn `Phase_identification` vì đó là nhãn protocol mô phỏng, không phải cảm biến thực tế.

Log:

```python
print(len(INSIDE_INPUT_COLS))
print(INSIDE_INPUT_COLS)
```

Kỳ vọng:

```text
16
('Temperature_In', 'Humidity_In', ...)
```

## 7. Viết hàm `log_step`

Trong `debug_walkthrough.py`, hàm này dùng để in dữ liệu sau từng bước:

```python
def log_step(title: str, value: Any, note: str = "", max_rows: int = 3) -> None:
    print("\n" + "=" * 88)
    print(f"STEP: {title}")
    if note:
        print(f"NOTE: {note}")
```

Nếu `value` là DataFrame:

```python
print(f"type=DataFrame shape={value.shape}")
print(f"columns={list(value.columns)}")
print(value.head(max_rows).to_string(index=False))
```

Nếu `value` là numpy array:

```python
print(f"type=ndarray shape={value.shape} dtype={value.dtype}")
print(preview)
```

Nếu `value` là dict:

```python
print(json.dumps(base.json_ready(value), ensure_ascii=False, indent=2))
```

Mục tiêu của logger:

- thấy dữ liệu có đúng shape không;
- thấy cột có tồn tại không;
- thấy output hàm có hợp lý không;
- bắt lỗi sớm trước khi train model.

## 8. Lỗi dễ gặp ở bước chuẩn bị

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| import sklearn lỗi | chưa cài dependency | chạy `pip install -r requirements.txt` |
| sai đường dẫn results | dùng path tương đối sai | dùng `Path(__file__).resolve()` |
| thiếu cột input | CSV thật thiếu cột | kiểm tra schema trước khi train |
| số input không khớp docs | quên cập nhật docs sau khi xóa feature | chạy `print(len(INSIDE_INPUT_COLS))` |
