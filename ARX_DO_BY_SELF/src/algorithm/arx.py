from __future__ import annotations

import numpy as np
import pandas as pd

from algorithm.specs import ArxSpec


# ARX dùng công thức:
# y(t) = a1*y(t-1) + ... + ana*y(t-na)
#      + b1*u1(t-nk) + ... + bnb*u1(t-nk-nb+1)
#      + ... cho toàn bộ input u
#      + bias


# Tính số dòng đầu tiên chưa dự đoán được vì thiếu dữ liệu quá khứ.
def max_lag(spec: ArxSpec) -> int:
    # Output cần y(t-na), input cần u(t-nk-nb+1), nên lấy lag lớn nhất.
    return max(spec.na, spec.nb + spec.nk - 1)


# Tạo X và y cho bài toán tuyến tính: y = X @ theta.
def build_arx_matrix(df_z: pd.DataFrame, spec: ArxSpec, input_cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    # y là Soil_Moisture đã chuẩn hóa.
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    lag = max_lag(spec)
    cols: list[np.ndarray] = []

    # Nhóm hệ số a: lấy các output quá khứ y(t-1), y(t-2), ..., y(t-na).
    for y_lag in range(1, spec.na + 1):
        cols.append(y[lag - y_lag : len(y) - y_lag])

    # Nhóm hệ số b: với mỗi input, lấy u(t-nk), u(t-nk-1), ..., đủ nb mẫu.
    for col in input_cols:
        values = df_z[col].to_numpy(dtype=float)
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            cols.append(values[lag - u_lag : len(values) - u_lag])

    # Cột 1 cuối cùng là bias để model tự học độ lệch nền.
    cols.append(np.ones(len(y) - lag))

    # X có dạng [y_lag, input_lag, bias], target là y(t) từ sau đoạn lag.
    return np.vstack(cols).T, y[lag:]


# Fit theta, tức toàn bộ hệ số a, b và bias của công thức ARX.
def fit_arx(df_train_z: pd.DataFrame, spec: ArxSpec, input_cols: tuple[str, ...]) -> np.ndarray:
    x_train, y_train = build_arx_matrix(df_train_z, spec, input_cols)

    # Nếu alpha = 0 thì giải least squares thường: min ||X@theta - y||^2.
    if spec.alpha <= 0.0:
        theta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
        return theta

    # Nếu alpha > 0 thì dùng ridge: min ||X@theta - y||^2 + alpha*||theta||^2.
    penalty = np.eye(x_train.shape[1], dtype=float)

    # Không phạt bias vì bias chỉ là độ lệch nền, không phải hệ số động học.
    penalty[-1, -1] = 0.0

    # Nghiệm ridge: theta = (X'X + alpha*I)^-1 X'y.
    lhs = x_train.T @ x_train + spec.alpha * penalty
    rhs = x_train.T @ y_train
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        # Nếu ma trận gần suy biến thì fallback sang least squares để vẫn có nghiệm.
        theta, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return theta


# Tính y_hat(t) từ công thức ARX tại đúng một thời điểm.
def predict_at(y_source: np.ndarray, inputs: list[np.ndarray], t: int, theta: np.ndarray, spec: ArxSpec) -> float:
    idx = 0
    y_next = 0.0

    # Cộng phần output quá khứ: a1*y(t-1) + ... + ana*y(t-na).
    for y_lag in range(1, spec.na + 1):
        y_next += theta[idx] * y_source[t - y_lag]
        idx += 1

    # Cộng phần input quá khứ cho từng biến điều khiển/môi trường.
    for values in inputs:
        for u_lag in range(spec.nk, spec.nk + spec.nb):
            y_next += theta[idx] * values[t - u_lag]
            idx += 1

    # Cộng bias cuối công thức.
    y_next += theta[idx]
    return float(y_next)


# One-step prediction: dự đoán y(t) bằng y quá khứ thật trong data.
def predict_one_step(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x, y_true = build_arx_matrix(df_z, spec, input_cols)

    # Vì dùng y thật ở các lag nên đây là cách đánh giá dễ nhất cho model.
    return np.clip(x @ theta, clip[0], clip[1]), y_true


# Free-run: chỉ dùng y thật để khởi động, sau đó y quá khứ là y model tự dự đoán.
def simulate_free_run(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    # y giữ giá trị thật để so sánh, y_sim là chuỗi mô phỏng của model.
    y = df_z["Soil_Moisture"].to_numpy(dtype=float).copy()
    y_sim = y.copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max_lag(spec)

    # Từ sau đoạn khởi động, mỗi y_sim(t) được đưa ngược lại làm lag cho bước sau.
    for t in range(lag, len(y)):
        y_sim[t] = float(np.clip(predict_at(y_sim, inputs, t, theta, spec), clip[0], clip[1]))

    return y_sim[lag:], y[lag:]


# Chunked simulation: mô phỏng từng đoạn ngắn, hết đoạn thì reset lại bằng y thật.
def simulate_chunked(
    df_z: pd.DataFrame,
    theta: np.ndarray,
    spec: ArxSpec,
    input_cols: tuple[str, ...],
    clip: tuple[float, float],
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Cách này mô phỏng các bài toán dự đoán 5 phút hoặc 20 phút.
    y = df_z["Soil_Moisture"].to_numpy(dtype=float)
    y_pred = y.copy()
    inputs = [df_z[col].to_numpy(dtype=float) for col in input_cols]
    lag = max_lag(spec)

    # Mỗi chunk bắt đầu từ y thật, rồi chạy tự do trong horizon_steps.
    for start in range(lag, len(y), horizon_steps):
        end = min(len(y), start + horizon_steps)
        y_work = y.copy()
        for t in range(start, end):
            # Trong cùng một chunk, dự đoán mới được dùng làm quá khứ cho bước sau.
            y_work[t] = float(np.clip(predict_at(y_work, inputs, t, theta, spec), clip[0], clip[1]))
            y_pred[t] = y_work[t]

    return y_pred[lag:], y[lag:]
