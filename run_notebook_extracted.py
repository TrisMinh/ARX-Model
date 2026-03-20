# Cài đặt (bỏ comment nếu cần)
# !pip install numpy pandas matplotlib scipy scikit-learn statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Matplotlib settings
plt.rcParams.update({
    'figure.figsize': (14, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

print('✅ Libraries imported successfully')
print(f'NumPy:  {np.__version__}')
print(f'Pandas: {pd.__version__}')

def generate_greenhouse_data(N=3000, T_s=60, seed=42):
    """
    Tạo dữ liệu nhà kính tổng hợp.
    
    Args:
        N:    số mẫu
        T_s:  sampling period (giây)
        seed: random seed
    Returns:
        df (DataFrame), true_params (dict)
    """
    np.random.seed(seed)
    t = np.arange(N)
    period_day = 24 * 3600 / T_s  # samples per day

    # ── TRUE PARAMETERS ──────────────────────────────────────
    TRUE = {
        'a1': 0.85, 'a2': -0.12,
        'b_temp': -0.8, 'b_humi': 0.3,
        'b_light': 0.02, 'b_drip': 5.0,
        'b_mist': 3.0, 'b_fan': -2.0,
    }

    # ── INPUTS ───────────────────────────────────────────────
    Temp  = 25 + 6*np.sin(2*np.pi*t/period_day) + np.random.normal(0, 1, N)
    Humi  = 70 + 15*np.cos(2*np.pi*t/period_day) + np.random.normal(0, 2, N)
    Light = np.clip(1200*np.sin(np.pi*(t % period_day)/period_day),
                    0, None) + np.random.normal(0, 30, N)
    Light = np.clip(Light, 0, 1500)

    # Actuators: random switching
    Drip = (np.random.random(N) > 0.85).astype(float)   # 15%
    Mist = (np.random.random(N) > 0.92).astype(float)   # 8%
    Fan  = (np.random.random(N) > 0.70).astype(float)   # 30%

    # ── OUTPUT: Soil Moisture ─────────────────────────────────
    y = np.zeros(N)
    y[0], y[1] = 65.0, 64.8

    for i in range(2, N):
        y[i] = (TRUE['a1'] * y[i-1] + TRUE['a2'] * y[i-2]
                + TRUE['b_temp']  * Temp[i-1]
                + TRUE['b_humi']  * Humi[i-1]
                + TRUE['b_light'] * Light[i-1]
                + TRUE['b_drip']  * Drip[i-1]
                + TRUE['b_mist']  * Mist[i-1]
                + TRUE['b_fan']   * Fan[i-1]
                + np.random.normal(0, 0.25))
        y[i] = np.clip(y[i], 0, 100)

    df = pd.DataFrame({
        'Timestamp':     pd.date_range('2025-01-01', periods=N, freq=f'{T_s}s'),
        'Soil_Moisture': y,
        'Temperature':   Temp,
        'Humidity':      Humi,
        'Light':         Light,
        'Drip':          Drip,
        'Mist':          Mist,
        'Fan':           Fan,
    })
    return df, TRUE


df, TRUE_PARAMS = generate_greenhouse_data(N=3000, T_s=60)
print(f'Dataset shape: {df.shape}')
print(f"Date range   : {df['Timestamp'].iloc[0]}  →  {df['Timestamp'].iloc[-1]}")
print('\nTrue parameters used for generation:')
for k, v in TRUE_PARAMS.items():
    print(f'  {k:12s} = {v:+.4f}')
df.head()

print('=== Descriptive Statistics ===')
print(df.describe().round(3).to_string())

# Visualize all signals
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
cols  = ['Soil_Moisture', 'Temperature', 'Humidity', 'Light', 'Drip', 'Mist', 'Fan']
colors= ['royalblue', 'tomato', 'mediumseagreen', 'darkorange',
          'mediumpurple', 'deepskyblue', 'coral']
axes_flat = axes.flatten()
for i, (col, c) in enumerate(zip(cols, colors)):
    axes_flat[i].plot(df[col].values[:500], color=c, linewidth=0.8)
    axes_flat[i].set_title(col, fontweight='bold')
    axes_flat[i].set_xlabel('Sample index')
axes_flat[-1].set_visible(False)
fig.suptitle('Greenhouse Sensor Signals (first 500 samples)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

from scipy.signal import savgol_filter

# ── CONFIG ────────────────────────────────────────────────────
FILTER_WINDOW  = 11   # nên là số lẻ; tăng lên để lọc mạnh hơn
FILTER_POLY    = 3    # bậc đa thức (3 hoặc 5 là phổ biến)
# ⚠️ QUAN TRỌNG: KHÔNG lọc Soil_Moisture (output y)!
# Lọc y trước khi fit sẽ làm mất noise dynamics thật → sigma² giả tạo nhỏ, FIT% ảo
CONTINUOUS_COLS = ['Temperature', 'Humidity', 'Light']  # CHỈ lọc inputs
BINARY_COLS     = ['Drip', 'Mist', 'Fan']   # không lọc binary signals

df_filtered = df.copy()

for col in CONTINUOUS_COLS:
    df_filtered[col] = savgol_filter(
        df[col].values,
        window_length=FILTER_WINDOW,
        polyorder=FILTER_POLY,
        mode='nearest'        # xử lý biên
    )

# ── VISUALIZE: Before vs After filtering ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
for ax, col in zip(axes.flatten(), CONTINUOUS_COLS):
    n_show = 300
    ax.plot(df[col].values[:n_show],          'gray', lw=0.8, alpha=0.6, label='Raw')
    ax.plot(df_filtered[col].values[:n_show], 'royalblue', lw=1.5, label='Filtered')
    ax.set_title(col, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('Sample index')

fig.suptitle(f'Savitzky-Golay Filter (window={FILTER_WINDOW}, poly={FILTER_POLY}) – first 300 samples',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# Calculate noise reduction
print('=== Noise Reduction ===')
for col in CONTINUOUS_COLS:
    noise_before = (df[col] - df[col].rolling(5).mean()).std()
    noise_after  = (df_filtered[col] - df_filtered[col].rolling(5).mean()).std()
    reduction = (1 - noise_after / noise_before) * 100 if noise_before > 0 else 0
    print(f'  {col:<15}  noise std: {noise_before:.4f} → {noise_after:.4f}  '
          f'({reduction:.1f}% reduction)')

print('\n✅ Filtering done. Using df_filtered for all downstream steps.')
df = df_filtered  # Replace raw df

# ── MIN-MAX SCALER ───────────────────────────────────────────
ALL_FEATURE_COLS = ['Soil_Moisture', 'Temperature', 'Humidity', 'Light']
# Binary cols (Drip, Mist, Fan) đã ở thang 0-1, không cần scale

# Fit on training portion (before split)
N_total = len(df)
TRAIN_RATIO = 0.70
n_train = int(N_total * TRAIN_RATIO)

scaler_min = df.iloc[:n_train][ALL_FEATURE_COLS].min()
scaler_max = df.iloc[:n_train][ALL_FEATURE_COLS].max()
scaler_range = scaler_max - scaler_min

def normalize(df_in, cols=ALL_FEATURE_COLS):
    df_out = df_in.copy()
    df_out[cols] = (df_in[cols] - scaler_min) / scaler_range
    return df_out

def denormalize_y(y_norm):
    """Chuyển soil moisture từ normalized về đơn vị %"""
    return y_norm * scaler_range['Soil_Moisture'] + scaler_min['Soil_Moisture']

df_norm = normalize(df)

# ── Show before/after ─────────────────────────────────────────
print('=== Scaling Info (fit on training set) ===')
for col in ALL_FEATURE_COLS:
    print(f'  {col:<15}  min={scaler_min[col]:.3f}  max={scaler_max[col]:.3f}  '
          f'range={scaler_range[col]:.3f}')

print('\n=== Normalized Statistics (should be ≈ [0, 1]) ===')
print(df_norm[ALL_FEATURE_COLS + BINARY_COLS].describe().round(3).to_string())

# Condition number comparison
from numpy.linalg import cond, matrix_rank

def quick_cond(df_in, na=2, nb=2, nk=1):
    n = int(len(df_in) * TRAIN_RATIO)
    df_tr = df_in.iloc[:n]
    X, _ = build_regression_matrix(df_tr, na, nb, nk) if 'build_regression_matrix' in dir() else (None, None)
    return cond(X.T @ X) if X is not None else float('nan')

print('\n💡 Condition number will be computed after build_regression_matrix is defined (Section 4).')
print('✅ Normalization done. Using df_norm for all downstream steps.')
df = df_norm  # Replace filtered df with normalized df

# ── CONFIG ──────────────────────────────────────────────────
NA = 2   # AR order
NB = 2   # input order
NK = 1   # input delay
TRAIN_RATIO = 0.70

# ── SPLIT ────────────────────────────────────────────────────
N_total = len(df)
n_train = int(N_total * TRAIN_RATIO)
df_train = df.iloc[:n_train].copy()
df_val   = df.iloc[n_train:].copy().reset_index(drop=True)

print(f'Total samples : {N_total}')
print(f'Training (DS1): {len(df_train)} samples  ({TRAIN_RATIO*100:.0f}%)')
print(f'Validation(DS2): {len(df_val)} samples  ({(1-TRAIN_RATIO)*100:.0f}%)')

INPUT_COLS = ['Temperature', 'Humidity', 'Light', 'Drip', 'Mist', 'Fan']
OUTPUT_COL = 'Soil_Moisture'


def build_regression_matrix(df, na=2, nb=2, nk=1,
                             input_cols=INPUT_COLS, output_col=OUTPUT_COL):
    """
    Xây dựng ma trận hồi quy X và vector Y cho ARX(na, nb, nk).

    Returns:
        X : (N_eff × n_params) regression matrix
        Y : (N_eff,)            output vector
    """
    y = df[output_col].values
    Us = [df[c].values for c in input_cols]
    n_inputs = len(Us)

    max_lag = max(na, nb + nk - 1)
    N       = len(y)
    N_eff   = N - max_lag
    n_params = na + n_inputs * nb + 1

    X = np.zeros((N_eff, n_params))
    Y = np.zeros(N_eff)

    for i in range(N_eff):
        t   = i + max_lag
        row = []
        # AR terms
        for lag in range(1, na + 1):
            row.append(y[t - lag])
        # eXogenous terms
        for u in Us:
            for lag in range(nk, nk + nb):
                row.append(u[t - lag])
        row.append(1.0)
        X[i] = row
        Y[i] = y[t]

    return X, Y


X_train, Y_train = build_regression_matrix(df_train, NA, NB, NK)
X_val,   Y_val   = build_regression_matrix(df_val,   NA, NB, NK)

print(f'X_train shape : {X_train.shape}  (N_eff × n_params)')
print(f'Y_train shape : {Y_train.shape}')
print(f'X_val   shape : {X_val.shape}')

# Check rank
rank = np.linalg.matrix_rank(X_train)
cond = np.linalg.cond(X_train.T @ X_train)
print(f'\nRank of X_train  : {rank} / {X_train.shape[1]}  (phải bằng nhau)')
print(f'Condition number : {cond:.2f}  (nên < 1e6)')

# Column names
PARAM_NAMES = (['a1','a2'] +
               [f'b_{c}_{l}' for c in INPUT_COLS for l in range(1, NB+1)] +
               ['intercept'])
print(f'\nParameter names ({len(PARAM_NAMES)} total):')
print(PARAM_NAMES)

# Correlation heatmap of regression matrix
import matplotlib.pyplot as plt

corr = np.corrcoef(X_train.T)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.abs(corr), cmap='RdYlGn_r', vmin=0, vmax=1)
ax.set_xticks(range(len(PARAM_NAMES)))
ax.set_yticks(range(len(PARAM_NAMES)))
ax.set_xticklabels(PARAM_NAMES, rotation=90, fontsize=9)
ax.set_yticklabels(PARAM_NAMES, fontsize=9)
plt.colorbar(im, ax=ax, label='|Correlation|')
ax.set_title('Correlation Matrix of Regressors (|corr|)', fontweight='bold')
plt.tight_layout()
plt.show()
print('High correlation (>0.8) may indicate multicollinearity issues.')

def estimate_ols(X, Y):
    """
    Ordinary Least Squares estimation.
    Returns: theta, cov_matrix, sigma2
    """
    # lstsq is numerically more stable than pinv
    theta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    N, k   = X.shape
    Y_pred = X @ theta
    resid  = Y - Y_pred
    sigma2 = np.dot(resid, resid) / (N - k)   # unbiased estimator
    cov    = sigma2 * np.linalg.pinv(X.T @ X)

    return theta, cov, sigma2


theta_hat, cov_hat, sigma2_hat = estimate_ols(X_train, Y_train)
std_hat = np.sqrt(np.diag(cov_hat))

# ── DENORMALIZE THETA về không gian gốc ─────────────────────
# Trong normalized space: y_n = a*y_n_lag + b_n*u_n_lag
# AR coefs (a1, a2) KHÔNG đổi (ratio y/y → scale cancel)
# b coefs: b_orig = b_norm * (y_range / u_range)
# Binary inputs (Drip, Mist, Fan) không được normalize → u_range = 1
# → b_binary_orig = b_norm * y_range

# Build scale vector cho từng tham số
theta_scale = []
for name in PARAM_NAMES:
    if name.startswith('a'):   # AR terms: scale = 1
        theta_scale.append(1.0)
    elif name == 'intercept':
        theta_scale.append(float(scaler_range['Soil_Moisture']))
    else:                       # b_InputName_lag
        # tên dạng 'b_Temperature_1' → input = 'Temperature'
        input_name = '_'.join(name.split('_')[1:-1])
        if input_name in scaler_range.index:  # input được normalize
            u_rng = float(scaler_range[input_name])
            theta_scale.append(float(scaler_range['Soil_Moisture']) / u_rng)
        else:                                  # binary: range = 1
            theta_scale.append(float(scaler_range['Soil_Moisture']))

theta_orig = theta_hat * np.array(theta_scale)  # theta trong đơn vị gốc (%)
sigma2_orig = sigma2_hat * float(scaler_range['Soil_Moisture'])**2  # var trong đơn vị %

print(f'Estimated noise σ (normalized): {np.sqrt(sigma2_hat):.6f}')
print(f'Estimated noise σ (original %): {np.sqrt(sigma2_orig):.4f} %  [true ≈ 0.25 %]\n')

print(f'{"Param":<15} {"θ̂_norm":>12}  {"θ̂_orig (%)":>13}  [TRUE (if known)]')
print('-' * 70)

# TRUE_PARAMS ở space gốc (%) – so sánh trực tiếp với theta_orig
true_ref = {
    'a1': TRUE_PARAMS['a1'],    'a2': TRUE_PARAMS['a2'],
    'b_Drip_1':  TRUE_PARAMS['b_drip'],
    'b_Temperature_1': TRUE_PARAMS['b_temp'],
}

for name, th_n, th_o in zip(PARAM_NAMES, theta_hat, theta_orig):
    ref   = true_ref.get(name, float('nan'))
    ref_s = f'{ref:+.4f}' if not np.isnan(ref) else '  —'
    match = ' ✅' if (not np.isnan(ref) and abs(th_o - ref) < 0.05) else (' ⚠️' if not np.isnan(ref) else '')
    print(f'{name:<15} {th_n:+12.6f}  {th_o:+13.6f}  {ref_s}{match}')

# Bar chart of estimated parameters
fig, ax = plt.subplots(figsize=(14, 5))
x_pos = np.arange(len(PARAM_NAMES))
colors_bar = ['royalblue' if v >= 0 else 'tomato' for v in theta_hat]
bars = ax.bar(x_pos, theta_hat, color=colors_bar, edgecolor='black', linewidth=0.7)
ax.errorbar(x_pos, theta_hat, yerr=1.96*std_hat, fmt='none',
            color='black', capsize=5, linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x_pos)
ax.set_xticklabels(PARAM_NAMES, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Parameter value θ̂')
ax.set_title('ARX(2,2,1) Estimated Parameters with 95% Confidence Intervals',
             fontweight='bold')
plt.tight_layout()
plt.show()

# ── ONE-STEP PREDICTION ──────────────────────────────────────
Y_train_pred = X_train @ theta_hat
Y_val_pred   = X_val   @ theta_hat


# ── FREE-RUN SIMULATION ──────────────────────────────────────
def simulate_arx(df_sim, theta, na=NA, nb=NB, nk=NK,
                 input_cols=INPUT_COLS, output_col=OUTPUT_COL):
    """
    Multi-step (free-run) simulation.  Uses predicted y values
    to feed back into the AR part.
    """
    y = df_sim[output_col].values.copy()  # ground truth (used as init)
    Us = [df_sim[c].values for c in input_cols]
    N  = len(y)
    max_lag = max(na, nb + nk - 1)
    y_sim = y.copy()

    for t in range(max_lag, N):
        row = []
        for lag in range(1, na + 1):
            row.append(y_sim[t - lag])   # use simulated (not true) y
        for u in Us:
            for lag in range(nk, nk + nb):
                row.append(u[t - lag])
        row.append(1.0)
        y_sim[t] = np.dot(row, theta)

    return y_sim[max_lag:], y[max_lag:]


Y_sim, Y_true_sim = simulate_arx(df_val, theta_hat)

print(f'One-step prediction on validation: {len(Y_val_pred)} samples')
print(f'Free-run simulation  on validation: {len(Y_sim)} samples')

# Plot prediction vs actual (validation set, first 300 samples)
n_plot = min(300, len(Y_val))
t_plot = np.arange(n_plot)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# One-step
axes[0].plot(t_plot, Y_val[:n_plot],      'b',  label='Actual',          lw=1.2)
axes[0].plot(t_plot, Y_val_pred[:n_plot], 'r--',label='1-step Prediction',lw=1.2)
axes[0].set_ylabel('Soil Moisture (%)')
axes[0].set_title('One-Step-Ahead Prediction – Validation Set', fontweight='bold')
axes[0].legend()

# Free-run
n_sim = min(n_plot, len(Y_sim))
axes[1].plot(np.arange(n_sim), Y_true_sim[:n_sim], 'b',  label='Actual',           lw=1.2)
axes[1].plot(np.arange(n_sim), Y_sim[:n_sim],      'm--',label='Simulation (free-run)', lw=1.2)
axes[1].set_xlabel('Sample index')
axes[1].set_ylabel('Soil Moisture (%)')
axes[1].set_title('Free-Run Simulation – Validation Set', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

def compute_metrics(Y_true, Y_pred, n_params):
    """Tính đầy đủ các metrics đánh giá mô hình."""
    N        = len(Y_true)
    resid    = Y_true - Y_pred
    ss_res   = np.dot(resid, resid)
    ss_tot   = np.dot(Y_true - Y_true.mean(), Y_true - Y_true.mean())
    sigma2   = ss_res / N

    rmse = np.sqrt(ss_res / N)
    mae  = np.mean(np.abs(resid))
    fit  = 100 * (1 - np.linalg.norm(resid) / np.linalg.norm(Y_true - Y_true.mean()))
    r2   = 1 - ss_res / ss_tot
    aic  = N * np.log(sigma2 + 1e-12) + 2 * n_params
    bic  = N * np.log(sigma2 + 1e-12) + n_params * np.log(N)

    return dict(RMSE=rmse, MAE=mae, FIT=fit, R2=r2, AIC=aic, BIC=bic)


n_params = len(theta_hat)

# ── DENORMALIZE về đơn vị % trước khi tính metrics ──────────────
# Lý do: OLS chạy trên normalized data → predictions ở normalized space
# Metrics (nhất là RMSE, MAE) phải ở original space để có nghĩa vật lý
Y_train_predO = denormalize_y(Y_train_pred)   # predictions → %
Y_train_trueO = denormalize_y(Y_train)         # ground truth → %
Y_val_predO   = denormalize_y(Y_val_pred)
Y_val_trueO   = denormalize_y(Y_val)
Y_sim_predO   = denormalize_y(Y_sim)
Y_sim_trueO   = denormalize_y(Y_true_sim)

print('=== Sanity check (should be in ~20-80% range) ===')
print(f'  Y_val_trueO: min={Y_val_trueO.min():.2f}%  max={Y_val_trueO.max():.2f}%')
print(f'  Y_val_predO: min={Y_val_predO.min():.2f}%  max={Y_val_predO.max():.2f}%')

m_train  = compute_metrics(Y_train_trueO, Y_train_predO, n_params)
m_val    = compute_metrics(Y_val_trueO,   Y_val_predO,   n_params)
m_sim    = compute_metrics(Y_sim_trueO,   Y_sim_predO,   n_params)

print(f"\n=== Metrics in ORIGINAL space (%) ===")
print(f"{'Metric':<8} | {'DS1 Train':>12} | {'DS2 Val (1-step)':>16} | {'DS2 Val (sim)':>14} | Target")
print('-' * 72)
targets = {'RMSE':'< 2.0 %', 'MAE':'< 1.5 %', 'FIT':'> 80 %',
           'R2':'> 0.90', 'AIC':'minimize', 'BIC':'minimize'}
for k in ['RMSE','MAE','FIT','R2','AIC','BIC']:
    unit = ' %' if k in ('FIT','RMSE','MAE') else ''
    print(f'{k:<8} | {m_train[k]:>12.4f}{unit} | {m_val[k]:>14.4f}{unit} | {m_sim[k]:>12.4f}{unit}  {targets[k]}')

# Radar chart – model performance summary
from matplotlib.patches import FancyArrowPatch

metrics_disp = {
    'FIT (%)':  m_val['FIT'] / 100,
    'R²':       m_val['R2'],
    '1-NRMSE':  1 - m_val['RMSE'] / (Y_val_trueO.max() - Y_val_trueO.min()),
    '1-MAE%':   1 - m_val['MAE']  / (Y_val_trueO.max() - Y_val_trueO.min()),
}

labels = list(metrics_disp.keys())
vals   = list(metrics_disp.values())
N_r    = len(labels)
angles = np.linspace(0, 2*np.pi, N_r, endpoint=False).tolist()
vals  += vals[:1];  angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, vals, 'o-', color='royalblue', linewidth=2)
ax.fill(angles, vals, alpha=0.25, color='royalblue')
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
ax.set_title('Model Performance Radar\n(Validation Set)', fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Residuals in ORIGINAL space (%) – đơn vị vật lý có nghĩa
residuals = Y_val_trueO - Y_val_predO
N_res = len(residuals)

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# 1. Residuals over time
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(residuals, 'g', linewidth=0.7, alpha=0.8)
ax1.axhline(0, color='k', linewidth=1, linestyle='--')
ax1.fill_between(range(N_res), residuals, 0, alpha=0.3, color='green')
ax1.set(xlabel='Sample', ylabel='Residual (%)', title='Residuals over Time')

# 2. Histogram + Normal fit
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(residuals, bins=50, density=True, color='purple', alpha=0.6, edgecolor='white')
x_r = np.linspace(residuals.min(), residuals.max(), 200)
mu_r, sd_r = residuals.mean(), residuals.std()
ax2.plot(x_r, stats.norm.pdf(x_r, mu_r, sd_r), 'r-', lw=2, label=f'N({mu_r:.3f},{sd_r:.3f})')
ax2.set(xlabel='Residual', ylabel='Density', title='Residual Distribution')
ax2.legend()

# 3. Q-Q plot
ax3 = fig.add_subplot(gs[1, 1])
stats.probplot(residuals, plot=ax3)
ax3.set_title('Q-Q Plot (Normality Check)')

# 4. Autocorrelation of residuals
ax4 = fig.add_subplot(gs[1, 2])
max_lag_acf = 40
acf = [np.corrcoef(residuals[:-k], residuals[k:])[0,1] for k in range(1, max_lag_acf+1)]
conf = 1.96 / np.sqrt(N_res)
ax4.bar(range(1, max_lag_acf+1), acf, color='steelblue', edgecolor='white', width=0.8)
ax4.axhline( conf, color='r', linestyle='--', lw=1.5, label='95% CI')
ax4.axhline(-conf, color='r', linestyle='--', lw=1.5)
ax4.set(xlabel='Lag', ylabel='ACF', title='Residual Autocorrelation')
ax4.legend()

plt.suptitle('Residual Analysis – Validation Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Statistical tests
from scipy.stats import shapiro, normaltest

print('=== STATISTICAL TESTS ON RESIDUALS ===')

# Normality
_, p_sw  = shapiro(residuals[:1000])
_, p_dag = normaltest(residuals)
print(f'\n1. Normality:')
print(f'   Shapiro-Wilk   p = {p_sw:.4f}  → {"NORMAL ✅" if p_sw>0.05 else "NOT normal ⚠️"}')
print(f'   D\'Agostino-K²  p = {p_dag:.4f}  → {"NORMAL ✅" if p_dag>0.05 else "NOT normal ⚠️"}')

# Ljung-Box test
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(residuals, lags=20, return_df=True)
    lb_fail = lb[lb['lb_pvalue'] < 0.05]
    print(f'\n2. Ljung-Box (H0: no autocorrelation):')
    if lb_fail.empty:
        print('   PASSED ✅ – residuals appear to be white noise')
    else:
        print(f'   FAILED ⚠️ – autocorrelation at lags: {lb_fail.index.tolist()}')
        print('   → Consider increasing na')
except ImportError:
    print('   statsmodels not installed – skip Ljung-Box')

# Cross-correlation with inputs
print(f'\n3. Cross-correlation residuals vs inputs (|cc| < 0.1 is OK):')
n_min = min(N_res, len(df_val) - max(NA, NB + NK - 1))
for col in INPUT_COLS:
    u_vals = df_val[col].values[max(NA, NB+NK-1): max(NA, NB+NK-1) + n_min]
    cc = np.corrcoef(residuals[:n_min], u_vals[:n_min])[0, 1]
    status = '✅' if abs(cc) < 0.1 else '⚠️'
    print(f'   corr(e, {col:<12}) = {cc:+.4f}  {status}')

print(f'\n4. Mean of residuals: {mu_r:.6f}  (target: ~0)')
print(f'5. Std  of residuals: {sd_r:.6f} %')

import itertools

def model_selection_search(df_tr, df_vl,
                            na_list=[1,2,3],
                            nb_list=[1,2,3],
                            nk_list=[1,2]):
    results = []
    combos = list(itertools.product(na_list, nb_list, nk_list))
    print(f'Testing {len(combos)} model structures...')

    for na, nb, nk in combos:
        try:
            Xt, Yt = build_regression_matrix(df_tr, na, nb, nk)
            Xv, Yv = build_regression_matrix(df_vl, na, nb, nk)
            th, _, _ = estimate_ols(Xt, Yt)
            Yv_p = Xv @ th
            n_p  = len(th)
            m    = compute_metrics(Yv, Yv_p, n_p)
            results.append({'na':na,'nb':nb,'nk':nk,'n_params':n_p, **m})
        except Exception:
            pass

    res_df = pd.DataFrame(results).sort_values('AIC').reset_index(drop=True)
    return res_df


sel_df = model_selection_search(df_train, df_val,
                                 na_list=[1,2,3],
                                 nb_list=[1,2,3],
                                 nk_list=[1,2])

print('\n=== Model Selection Results (sorted by AIC) ===')
display_cols = ['na','nb','nk','n_params','RMSE','FIT','R2','AIC','BIC']
print(sel_df[display_cols].head(10).to_string(index=False, float_format='{:.4f}'.format))
print(f'\n✅ Best model by AIC: ARX({sel_df.iloc[0].na:.0f}, '
      f'{sel_df.iloc[0].nb:.0f}, {sel_df.iloc[0].nk:.0f})')

# Pareto plot: FIT vs n_params
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc = axes[0].scatter(sel_df['n_params'], sel_df['FIT'], c=sel_df['AIC'],
                      cmap='RdYlGn_r', s=80, edgecolors='black', linewidth=0.5)
plt.colorbar(sc, ax=axes[0], label='AIC')
axes[0].set(xlabel='Number of parameters', ylabel='FIT (%)',
             title='Pareto: Complexity vs Accuracy')
# Annotate best
best = sel_df.iloc[0]
axes[0].annotate(f"ARX({best.na:.0f},{best.nb:.0f},{best.nk:.0f})",
                  xy=(best.n_params, best.FIT), fontsize=10, color='darkgreen',
                  xytext=(best.n_params+0.5, best.FIT-2))

# AIC bar for top-10
top10 = sel_df.head(10)
labels_10 = [f"({r.na:.0f},{r.nb:.0f},{r.nk:.0f})" for _, r in top10.iterrows()]
colors_10 = plt.cm.RdYlGn_r(np.linspace(0, 0.8, len(top10)))
axes[1].barh(labels_10[::-1], top10['AIC'].values[::-1],
              color=colors_10, edgecolor='black', linewidth=0.5)
axes[1].set(xlabel='AIC', title='Top-10 Models by AIC (lower = better)')

plt.tight_layout()
plt.show()

# Save estimated parameters to file for later use
import json

# ── Denormalize metrics về không gian gốc ─────────────────
# 1-step prediction trong original space (denormalize predictions)
Y_train_orig = denormalize_y(Y_train_pred)
Y_train_true_orig = denormalize_y(Y_train)
Y_val_orig   = denormalize_y(Y_val_pred)
Y_val_true_orig = denormalize_y(Y_val)
Y_sim_orig   = denormalize_y(Y_sim)
Y_sim_true_orig = denormalize_y(Y_true_sim)

m_train_orig = compute_metrics(Y_train_true_orig, Y_train_orig, n_params)
m_val_orig   = compute_metrics(Y_val_true_orig,   Y_val_orig,   n_params)
m_sim_orig   = compute_metrics(Y_sim_true_orig,   Y_sim_orig,   n_params)

print('=== Metrics in ORIGINAL space (%) ===')
print(f"{'Metric':<8} | {'DS1 Train':>12} | {'DS2 Val (1-step)':>16} | {'DS2 Val (sim)':>14}")
print('-' * 60)
for k in ['RMSE','MAE','FIT','R2']:
    print(f'{k:<8} | {m_train_orig[k]:>12.4f} | {m_val_orig[k]:>16.4f} | {m_sim_orig[k]:>14.4f}')
print()
print(f'σ (train residuals original): {np.sqrt(m_train_orig["RMSE"]**2):.4f} %  [true ≈ 0.25 %]')

model_config = {
    'model': 'ARX',
    'na': NA, 'nb': NB, 'nk': NK,
    'n_params': int(n_params),
    'param_names': PARAM_NAMES,
    # Lưu theta trong ORIGINAL space (đơn vị %)
    'theta_hat': theta_orig.tolist(),
    # Lưu sigma2 trong original space (đơn vị %²)
    'sigma2': float(sigma2_orig),
    # Cũng lưu normalized version để dùng trong prediction (cần normalize input trước)
    'theta_hat_normalized': theta_hat.tolist(),
    'scaler_min': scaler_min.to_dict(),
    'scaler_max': scaler_max.to_dict(),
    # Metrics trong original space
    'metrics_train': m_train_orig,
    'metrics_val':   m_val_orig,
    'metrics_sim':   m_sim_orig,
    'input_cols': INPUT_COLS,
    'output_col': OUTPUT_COL,
}

with open('arx_model.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print('\n✅ Model saved to arx_model.json (theta in ORIGINAL % space)')
print(json.dumps({k: model_config[k] for k in ['model','na','nb','nk','n_params','sigma2']}, indent=2))

