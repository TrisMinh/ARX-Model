"""
Debug ARX: reproduce notebook data and sweep model orders.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

TRUE = {
    'a1': 0.85, 'a2': -0.12,
    'b_temp': -0.8, 'b_humi': 0.3, 'b_light': 0.02,
    'b_drip': 5.0, 'b_mist': 3.0, 'b_fan': -2.0,
}

N = 3000; period_day = 1440
t     = np.arange(N)
Temp  = 25 + 6*np.sin(2*np.pi*t/period_day) + np.random.normal(0, 1, N)
Humi  = 70 + 15*np.cos(2*np.pi*t/period_day) + np.random.normal(0, 2, N)
Light = np.clip(1200*np.sin(np.pi*(t % period_day)/period_day), 0, None) + np.random.normal(0, 30, N)
Light = np.clip(Light, 0, 1500)
Drip  = (np.random.random(N) > 0.85).astype(float)
Mist  = (np.random.random(N) > 0.92).astype(float)
Fan   = (np.random.random(N) > 0.70).astype(float)

y = np.zeros(N)
y[0], y[1] = 65.0, 64.8
for i in range(2, N):
    y[i] = (TRUE['a1']*y[i-1] + TRUE['a2']*y[i-2]
            + TRUE['b_temp']*Temp[i-1]  + TRUE['b_humi']*Humi[i-1]
            + TRUE['b_light']*Light[i-1] + TRUE['b_drip']*Drip[i-1]
            + TRUE['b_mist']*Mist[i-1]  + TRUE['b_fan']*Fan[i-1]
            + np.random.normal(0, 0.25))
    y[i] = np.clip(y[i], 0, 100)

df = pd.DataFrame({'Soil_Moisture': y, 'Temperature': Temp, 'Humidity': Humi,
                   'Light': Light, 'Drip': Drip, 'Mist': Mist, 'Fan': Fan})

print(f"y: mean={y.mean():.1f}%  std={y.std():.1f}%  range=[{y.min():.1f}, {y.max():.1f}]")

ALL_FEATURE_COLS = ['Soil_Moisture', 'Temperature', 'Humidity', 'Light']
n_train = int(N * 0.70)

scaler_min   = df.iloc[:n_train][ALL_FEATURE_COLS].min()
scaler_max   = df.iloc[:n_train][ALL_FEATURE_COLS].max()
scaler_range = scaler_max - scaler_min

def normalize(df_in):
    df_out = df_in.copy()
    df_out[ALL_FEATURE_COLS] = (df_in[ALL_FEATURE_COLS] - scaler_min) / scaler_range
    return df_out

df_norm = normalize(df)
INPUT_COLS = ['Temperature', 'Humidity', 'Light', 'Drip', 'Mist', 'Fan']
OUTPUT_COL = 'Soil_Moisture'

def build(df_in, na, nb, nk):
    y_v  = df_in[OUTPUT_COL].values
    Us   = [df_in[c].values for c in INPUT_COLS]
    max_lag = max(na, nb + nk - 1)
    rows = []
    for t_i in range(max_lag, len(y_v)):
        row = [y_v[t_i - lag] for lag in range(1, na + 1)]
        for u in Us:
            for lag in range(nk, nk + nb):
                row.append(u[t_i - lag])
        rows.append(row)
    return np.array(rows), y_v[max_lag:]

def fit_eval(Na, Nb, Nk):
    df_tr = df_norm.iloc[:n_train].copy()
    df_vl = df_norm.iloc[n_train:].copy().reset_index(drop=True)
    X_tr, Y_tr = build(df_tr, Na, Nb, Nk)
    X_vl, Y_vl = build(df_vl, Na, Nb, Nk)
    theta, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)

    def fit_pct(Yt, Yp):
        r = Yt - Yp
        return 100*(1 - np.linalg.norm(r)/np.linalg.norm(Yt - Yt.mean()))

    def r2(Yt, Yp):
        return 1 - np.sum((Yt-Yp)**2)/np.sum((Yt - Yt.mean())**2)

    ar_poly = np.array([1.0] + [-theta[i] for i in range(Na)])
    poles   = np.roots(ar_poly)
    stable  = all(abs(p) < 1.0 for p in poles)
    print(f"  na={Na} nb={Nb} nk={Nk} | FIT_tr={fit_eval.__wrapped__ if False else fit_pct(Y_tr,X_tr@theta):6.1f}%"
          f"  FIT_vl={fit_pct(Y_vl,X_vl@theta):6.1f}%  R2_vl={r2(Y_vl,X_vl@theta):.3f}"
          f"  poles={np.abs(poles).round(3).tolist()}  stable={stable}")
    return theta

print("\n=== Sweep model orders ===")
for na in [1, 2]:
    for nb in [1, 2]:
        fit_eval(na, nb, 1)

print(f"\n=== Distribution shift ===")
print(f"  y_train: mean={y[:n_train].mean():.2f}  std={y[:n_train].std():.2f}")
print(f"  y_val:   mean={y[n_train:].mean():.2f}  std={y[n_train:].std():.2f}")

print(f"\n=== Sanity: FIT in original space MUST equal FIT in norm space ===")
df_tr = df_norm.iloc[:n_train].copy()
df_vl = df_norm.iloc[n_train:].copy().reset_index(drop=True)
X_tr, Y_tr = build(df_tr, 2, 2, 1)
X_vl, Y_vl = build(df_vl, 2, 2, 1)
theta, _, _, _ = np.linalg.lstsq(X_tr, Y_tr, rcond=None)

def denormalize_y(yn):
    return yn * scaler_range['Soil_Moisture'] + scaler_min['Soil_Moisture']

Y_vl_O   = denormalize_y(Y_vl)
Y_pred_O = denormalize_y(X_vl @ theta)

def fit_pct(Yt, Yp):
    r = Yt - Yp
    return 100*(1 - np.linalg.norm(r)/np.linalg.norm(Yt - Yt.mean()))

fit_norm = fit_pct(Y_vl, X_vl @ theta)
fit_orig = fit_pct(Y_vl_O, Y_pred_O)
print(f"  FIT_norm_val = {fit_norm:.2f}%  |  FIT_orig_val = {fit_orig:.2f}%")
print(f"  (difference should be ~0, proving FIT is scale-invariant)")

RMSE_norm = np.sqrt(np.mean((Y_vl - X_vl@theta)**2))
RMSE_orig = np.sqrt(np.mean((Y_vl_O - Y_pred_O)**2))
print(f"\n  RMSE_norm = {RMSE_norm:.6f} (normalized units)")
print(f"  RMSE_orig = {RMSE_orig:.4f} % (physical units)")
print(f"  Ratio RMSE_orig/RMSE_norm = {RMSE_orig/RMSE_norm:.2f} ≈ scaler_range = {scaler_range['Soil_Moisture']:.2f}")
