# Ariel Data Challenge 2025 — Tổng thể kế hoạch mô hình hóa

## 1. Định hướng tổng thể

Bài toán **Ariel Data Challenge 2025** là bài toán **multi-target probabilistic regression**: từ dữ liệu quan sát nhiễu của kính thiên văn Ariel, dự đoán phổ khí quyển ngoại hành tinh và độ bất định tương ứng.

Mục tiêu đầu ra là phổ gồm **283 giá trị** xấp xỉ:

$$
\left(\frac{R_p}{R_s}\right)^2
$$

kèm theo **283 giá trị uncertainty / sigma**.

Các solution top cho thấy yếu tố quan trọng không chỉ là chọn model mạnh, mà là kết hợp:

- signal processing;
- physical prior;
- light curve fitting;
- physics-based feature engineering;
- uncertainty calibration.

Do đó, hướng chính được chọn là:

> **Bayesian Ridge Regression + Physics-based Feature Engineering + Target PCA + Sigma Calibration**

Pipeline tổng quát:

```text
Raw AIRS / FGS signals
→ detector calibration
→ light curve extraction
→ transit boundary detection
→ physics-based feature engineering
→ Bayesian Ridge / comparative ML models
→ uncertainty calibration
→ final spectrum + sigma
```

---

## 2. Pipeline tổng thể đề xuất

### Stage 1 — Data loading

Input chính gồm:

```text
AIRS-CH0 signal
FGS1 signal
calibration files
adc_info
star_info
train target spectrum
```

Vai trò của từng thành phần:

| Thành phần | Vai trò |
|---|---|
| AIRS-CH0 | Cung cấp thông tin phổ theo wavelength |
| FGS1 | Cung cấp white-light signal để bắt timing, baseline và transit shape |
| calibration files | Hiệu chỉnh detector, dead pixel, dark current, flat field |
| adc_info | Gain / offset để hiệu chỉnh raw signal |
| star_info | Thông tin sao chủ, hỗ trợ suy luận transit depth |
| train target | Ground truth spectrum gồm 283 chiều |

---

### Stage 2 — Detector calibration

Mục tiêu là loại bỏ nhiễu và sai lệch từ detector trước khi trích xuất light curve.

Pipeline calibration:

```text
raw signal
→ ADC correction
→ bad pixel masking
→ dark current subtraction
→ flat-field correction
→ linearity correction nếu có
→ correlated double sampling
→ temporal binning
```

#### 2.1. ADC correction

Dùng `gain` và `offset` từ `adc_info`:

$$
S_{\text{corr}} = (S_{\text{raw}} - \text{offset}) \times \text{gain}
$$

Mục tiêu: đưa raw digital counts về scale nhất quán hơn.

#### 2.2. Bad pixel masking

Dùng các file calibration như:

```text
dead.parquet
dark.parquet
flat.parquet
```

Cách xử lý:

- pixel chết: thay bằng median của pixel lân cận hoặc bỏ qua khi tổng hợp;
- hot pixel: phát hiện bằng sigma clipping theo thời gian;
- frame bất thường: loại hoặc giảm trọng số nếu tổng flux quá lệch.

#### 2.3. Dark current subtraction

$$
S_{\text{dark-corrected}} = S_{\text{corr}} - D
$$

#### 2.4. Flat-field correction

$$
S_{\text{flat-corrected}} = \frac{S_{\text{dark-corrected}}}{F}
$$

#### 2.5. Correlated Double Sampling — CDS

Giảm read noise bằng cách lấy hiệu giữa các lần đọc liên tiếp:

$$
S^{\text{CDS}}_t = S_{t+1} - S_t
$$

hoặc theo cặp frame:

$$
S^{\text{CDS}}_k = S_{2k+1} - S_{2k}
$$

#### 2.6. Temporal binning

Raw time-series rất dài, nên cần giảm số timestep:

```text
AIRS frames → 300 hoặc 500 bins
FGS1 frames → 300 hoặc 500 bins
```

Binning có thể dùng mean hoặc median:

$$
\bar{S}_b = \frac{1}{|B_b|}\sum_{t \in B_b} S_t
$$

---

### Stage 3 — Light curve extraction

Sau calibration, chuyển detector tensor thành light curve.

#### 3.1. AIRS light curve theo wavelength

Collapse theo chiều spatial:

$$
\text{LC}_{\lambda}(t) = \sum_x S(t, x, \lambda)
$$

Output:

```text
AIRS_LC shape = [time_bins, n_wavelengths]
```

#### 3.2. FGS1 white light curve

Collapse toàn ảnh:

$$
\text{LC}_{\text{FGS}}(t) = \sum_{x,y} S(t,x,y)
$$

Output:

```text
FGS_LC shape = [time_bins]
```

---

### Stage 4 — Normalize, detrend, smooth

#### 4.1. Normalize theo out-of-transit baseline

$$
\text{LC}_{\text{norm}}(t,\lambda) = \frac{\text{LC}(t,\lambda)}{\text{median}(\text{LC}_{\text{OOT},\lambda})}
$$

Nếu chưa biết vùng ngoài transit, có thể dùng percentile cao hoặc vùng đầu/cuối observation.

#### 4.2. Polynomial detrending

Fit baseline ngoài transit:

$$
b_\lambda(t) = a_0 + a_1t + a_2t^2
$$

Sau đó:

$$
\text{LC}_{\text{detrended}}(t,\lambda) = \frac{\text{LC}_{\text{norm}}(t,\lambda)}{b_\lambda(t)}
$$

Nên thử polynomial degree 1, 2, 3. Không nên dùng degree quá cao vì có thể fit mất transit dip.

#### 4.3. Moving average smoothing nhẹ

$$
\tilde{\text{LC}}(t) = \frac{1}{2k+1}\sum_{i=-k}^{k} \text{LC}(t+i)
$$

Gợi ý:

```text
window = 3, 5, 7
```

Không nên smooth quá mạnh vì có thể làm mờ ingress/egress.

---

### Stage 5 — Transit boundary detection

Cần xác định các vùng:

```text
out-of-transit left
ingress
in-transit
egress
out-of-transit right
```

Cách làm đề xuất: dùng FGS1 vì đây là white-light signal, thường có SNR tổng thể tốt hơn.

Các bước:

1. Smooth FGS light curve.
2. Tính đạo hàm bậc nhất:

$$
d_1(t) = \text{LC}(t+1) - \text{LC}(t)
$$

3. Tính đạo hàm bậc hai:

$$
d_2(t) = d_1(t+1) - d_1(t)
$$

4. Dùng các điểm biến thiên mạnh để xác định ingress/egress.
5. Áp dụng timing tìm được từ FGS1 sang AIRS.

Nếu dữ liệu đã được align tốt, có thể dùng fixed window ở giữa observation như một baseline đơn giản.

---

## 3. Physics-based feature engineering

Đây là phần quan trọng nhất của pipeline.

### 3.1. Transit depth features

Với mỗi wavelength:

$$
d_\lambda = 1 - \frac{\text{mean}(\text{LC}_\lambda^{\text{IN}})}{\text{mean}(\text{LC}_\lambda^{\text{OOT}})}
$$

Vì target gần với:

$$
d_\lambda \approx \left(\frac{R_p}{R_s}\right)^2
$$

Nên transit depth là nhóm feature quan trọng nhất.

Các biến thể nên tạo:

```text
depth_mean
depth_median
depth_min
depth_mid_transit
depth_percentile_5
depth_percentile_10
depth_weighted_by_noise
```

---

### 3.2. Multi-scale spectral features

Tạo feature theo các window wavelength:

```text
bin size = 1, 2, 4, 8, 16, 32, 64
```

Với mỗi bin:

```text
mean depth
median depth
std depth
slope across wavelength
local curvature
```

Lý do: phổ khí quyển thường có tính liên tục theo wavelength. Multi-scale features giúp giảm nhiễu và giữ cấu trúc phổ.

---

### 3.3. Transit shape features

Từ FGS1 và AIRS white curve:

```text
ingress_slope
egress_slope
transit_duration
mid_transit_flux
symmetry
curvature_bottom
oot_left_trend
oot_right_trend
baseline_drift
```

Ví dụ:

$$
s_{\text{ingress}} = \frac{\text{LC}(t_{\text{in}}) - \text{LC}(t_{\text{start}})}{t_{\text{in}} - t_{\text{start}}}
$$

Symmetry:

$$
\text{symmetry} = \left| \text{LC}_{\text{left half IN}} - \text{LC}_{\text{right half IN}} \right|
$$

---

### 3.4. Noise / uncertainty proxy features

Vì metric yêu cầu sigma, cần tạo feature đo độ tin cậy:

```text
oot_std
in_transit_std
residual_std_after_detrending
snr_depth
bad_pixel_count
flat_field_variance
dark_current_level
cds_noise_proxy
```

SNR:

$$
\text{SNR}_\lambda = \frac{d_\lambda}{\text{std}(\text{LC}^{\text{OOT}}_\lambda)}
$$

---

### 3.5. Stellar metadata features

Thêm thông tin sao chủ:

```text
Rs
Ms
Ts
logg
period
```

Có thể thêm log-transform:

```text
log(Rs)
log(Ms)
log(Ts)
```

và interaction:

```text
depth × Rs
depth × Ts
depth × logg
```

---

## 4. Model chính: Bayesian Ridge + Physics-based features

### 4.1. Lý do chọn Bayesian Ridge

Bayesian Ridge phù hợp vì:

- dữ liệu train không quá lớn;
- feature có thể tương quan mạnh;
- cần regularization;
- cần uncertainty;
- dễ giải thích trong báo cáo;
- phù hợp với metric probabilistic như Gaussian Log-Likelihood.

Đây là hướng kết hợp tinh thần của các top solution:

```text
Bayesian uncertainty từ rank 1
feature engineering từ rank 2
physics-based feature từ rank 4
physical fitting + ML correction từ rank 5
```

---

### 4.2. Target PCA

Thay vì train trực tiếp 283 targets, nên dùng PCA trên target spectrum:

$$
Y \in \mathbb{R}^{n \times 283}
$$

$$
Z = \text{PCA}(Y)
$$

Train Bayesian Ridge để dự đoán PCA coefficients:

$$
z_k = Xw_k + \epsilon
$$

Sau đó inverse transform:

$$
\hat{Y} = \text{PCA}^{-1}(\hat{Z})
$$

Ưu điểm:

```text
giảm số model từ 283 xuống 15–50
giữ tính mượt của spectrum
giảm overfitting
khai thác correlation giữa wavelengths
```

Khuyến nghị thử:

```text
n_components = 20, 30, 40
```

---

### 4.3. Bayesian Ridge formulation

Với mỗi PCA component:

$$
z_k = Xw_k + \epsilon
$$

Bayesian Ridge đặt prior:

$$
w_k \sim \mathcal{N}(0, \lambda^{-1}I)
$$

Noise:

$$
\epsilon \sim \mathcal{N}(0, \alpha^{-1})
$$

Model trả về:

```text
mean prediction
predictive standard deviation
```

---

### 4.4. Uncertainty estimation

Bayesian Ridge trả về uncertainty trong PCA space. Chuyển về wavelength space:

$$
\sigma^2_{Y,j} = \sum_k W_{k,j}^2 \sigma^2_{Z,k}
$$

Sau đó cộng residual RMSE từ validation:

$$
\sigma_{\text{final},j} = \sqrt{\sigma^2_{\text{BR},j} + \text{RMSE}^2_{\text{val},j}}
$$

Cuối cùng calibrate bằng hệ số scale:

$$
\sigma' = s \cdot \sigma_{\text{final}}
$$

Tối ưu $s$ trên validation theo Gaussian NLL / GLL.

---

## 5. Năm model ML thuần để so sánh

### Model 1 — Ridge Regression

#### Vai trò

Baseline tuyến tính mạnh.

#### Input

Physics-based features giống model chính.

#### Output

Spectrum trực tiếp hoặc PCA coefficients.

#### Ưu điểm

```text
nhanh
ổn định
dễ giải thích
ít overfit
```

#### Nhược điểm

```text
không có uncertainty tự nhiên tốt như Bayesian Ridge
khó bắt nonlinear interaction
```

#### Mục đích so sánh

Cho thấy lợi ích của Bayesian uncertainty so với regularized linear regression thông thường.

---

### Model 2 — Bayesian Ridge Regression

#### Vai trò

Model chính / best pure ML model.

#### Input

Physics-based features + target PCA.

#### Output

Mean spectrum + sigma.

#### Ưu điểm

```text
regularization tự nhiên
hợp dữ liệu nhỏ
có predictive uncertainty
giải thích tốt
```

#### Nhược điểm

```text
chủ yếu tuyến tính
phụ thuộc nhiều vào feature engineering
```

---

### Model 3 — Kernel Ridge Regression

#### Vai trò

Nonlinear baseline nhẹ.

Kernel nên thử:

```text
RBF kernel
polynomial kernel degree 2
laplacian kernel
```

#### Ý tưởng

\[
f(x) = \sum_i \alpha_i K(x, x_i)
\]

#### Ưu điểm

```text
bắt nonlinear pattern
vẫn tương đối ổn định
hợp dataset nhỏ-vừa
```

#### Nhược điểm

```text
khó scale nếu sample lớn
không có uncertainty tự nhiên
nhạy với kernel và gamma
```

---

### Model 4 — ExtraTrees / Random Forest Regressor

#### Vai trò

Tree-based nonlinear baseline.

#### Ưu điểm

```text
robust với outlier
ít cần scale feature
bắt interaction phi tuyến
có feature importance
```

#### Nhược điểm

```text
prediction theo wavelength có thể không mượt
không extrapolate tốt
uncertainty chỉ xấp xỉ qua ensemble variance
```

---

### Model 5 — LightGBM / XGBoost Regressor

#### Vai trò

Model tabular nonlinear mạnh.

Cách train:

```text
train 283 models riêng
hoặc train PCA components
```

Khuyến nghị: train PCA components để giảm overfitting.

#### Ưu điểm

```text
bắt nonlinear interaction rất tốt
mạnh trên tabular features
có thể dùng làm residual correction cho Bayesian Ridge
```

#### Nhược điểm

```text
dễ overfit vì số planet không quá lớn
cần tune kỹ
uncertainty phải tự thiết kế
```

#### Vai trò tốt nhất

Dùng làm residual correction:

$$
\hat{Y}_{\text{final}} = \hat{Y}_{\text{BR}} + \eta \cdot \hat{R}_{\text{LGBM}}
$$

với:

```text
eta = 0.2–0.5
```

---

## 6. Một vài model deep learning thuần để so sánh

Các model deep learning dùng để làm comparative study, không nhất thiết là model chính.

---

### DL Model 1 — 1D CNN trên light curve

#### Input

```text
AIRS_LC: [time_bins, wavelengths]
FGS_LC: [time_bins, 1]
```

Có thể dùng 2 branch:

```text
AIRS branch: Conv1D theo time
FGS branch: Conv1D theo time
concat + MLP
output: 283 mean + 283 log_sigma
```

#### Ưu điểm

```text
học local temporal pattern
bắt ingress/egress tự động
nhanh hơn Transformer
```

#### Nhược điểm

```text
cần nhiều dữ liệu
dễ overfit
khó giải thích hơn Bayesian Ridge
```

---

### DL Model 2 — Temporal CNN / TCN

#### Ý tưởng

Dùng dilated convolution để bắt pattern dài theo thời gian:

```text
Conv1D dilation = 1, 2, 4, 8, 16
```

#### Ưu điểm

```text
bắt dependency dài tốt hơn CNN thường
train ổn định hơn Transformer
phù hợp time-series
```

#### Nhược điểm

```text
vẫn cần regularization mạnh
không có physical prior rõ
```

---

### DL Model 3 — LSTM / GRU

#### Input

```text
sequence = [time_bins, channels]
```

#### Output

```text
spectrum mean
spectrum log_sigma
```

#### Ưu điểm

```text
phù hợp dữ liệu tuần tự
dễ implement
```

#### Nhược điểm

```text
train chậm
khó bắt long-range hơn TCN/Transformer
có thể kém hiệu quả với nhiều wavelength channels
```

---

### DL Model 4 — Transformer Encoder

#### Input

Mỗi timestep là một token:

$$
x_t = [\text{LC}_{\text{FGS}}(t), \text{LC}_{\text{AIRS},1}(t), \dots, \text{LC}_{\text{AIRS},\lambda}(t)]
$$

#### Ưu điểm

```text
bắt global temporal dependency
attention có thể học vùng transit quan trọng
```

#### Nhược điểm

```text
rất dễ overfit
cần nhiều dữ liệu
tốn compute
khó justify nếu dataset nhỏ
```

---

### DL Model 5 — Autoencoder + MLP

#### Ý tưởng

```text
light curve
→ encoder
→ latent vector
→ regression head
→ 283 mean + 283 sigma
```

#### Ưu điểm

```text
giảm chiều dữ liệu
học representation compact
có thể pretrain reconstruction
```

#### Nhược điểm

```text
latent representation có thể không tối ưu cho target
khó kiểm soát physical meaning
```

---

## 7. Bảng so sánh tổng hợp

| Nhóm | Model | Input chính | Output | Ưu điểm | Nhược điểm | Vai trò |
|---|---|---|---|---|---|---|
| ML thuần | Ridge | Physics features | Spectrum/PCA | Nhanh, ổn định | Không có uncertainty tốt | Baseline |
| ML thuần | Bayesian Ridge | Physics features | Mean + sigma | Hợp metric, dễ giải thích | Tuyến tính | Model chính |
| ML thuần | Kernel Ridge | Physics features | Spectrum/PCA | Bắt nonlinear nhẹ | Tune kernel khó | Nonlinear baseline |
| ML thuần | ExtraTrees/RF | Physics features | Spectrum/PCA | Robust, feature importance | Spectrum kém mượt | Tree baseline |
| ML thuần | LightGBM/XGBoost | Physics features | Spectrum/PCA/residual | Mạnh tabular | Dễ overfit | Residual correction |
| Deep | 1D CNN | Light curve | Mean + sigma | Học pattern tự động | Dễ overfit | DL baseline |
| Deep | TCN | Light curve | Mean + sigma | Bắt temporal dài | Ít physical prior | DL baseline |
| Deep | LSTM/GRU | Light curve | Mean + sigma | Dễ hiểu cho sequence | Chậm, kém scale | Sequence baseline |
| Deep | Transformer | Light curve tokens | Mean + sigma | Global attention | Rất dễ overfit | Advanced baseline |
| Deep | Autoencoder + MLP | Light curve latent | Mean + sigma | Giảm chiều tốt | Latent khó giải thích | Representation baseline |

---

## 8. Plan thực nghiệm đề xuất

### Experiment 0 — Data sanity check

Mục tiêu:

```text
kiểm tra shape
kiểm tra missing value
kiểm tra outlier planet
plot vài light curves
plot vài target spectra
```

Output báo cáo:

```text
sample AIRS light curve
sample FGS light curve
sample target spectrum
noise distribution
```

---

### Experiment 1 — Simple physics baseline

Feature:

```text
basic transit depth
FGS depth
star_info
```

Model:

```text
Ridge Regression
```

Mục tiêu:

```text
tạo baseline nhanh
kiểm tra feature có signal không
```

---

### Experiment 2 — Full physics-based features

Feature:

```text
depth variants
multi-scale wavelength features
shape features
noise features
stellar metadata
```

Models:

```text
Ridge
Bayesian Ridge
Kernel Ridge
ExtraTrees
LightGBM
```

Mục tiêu:

```text
so sánh ML thuần
chọn model mạnh nhất
```

---

### Experiment 3 — PCA target modeling

So sánh:

```text
direct 283-target prediction
vs
PCA target prediction
```

Models:

```text
Ridge + PCA
Bayesian Ridge + PCA
LightGBM + PCA
```

Mục tiêu:

```text
kiểm tra PCA có làm spectrum mượt hơn và giảm overfit không
```

---

### Experiment 4 — Uncertainty calibration

So sánh sigma:

```text
constant sigma
validation RMSE sigma
Bayesian predictive sigma
Bayesian sigma + residual RMSE
ensemble variance sigma
```

Metric:

```text
Gaussian NLL / GLL trên validation
```

Mục tiêu:

```text
chọn cách dự đoán sigma tốt nhất
```

---

### Experiment 5 — Deep learning comparison

Input:

```text
calibrated + binned light curves
```

Models:

```text
1D CNN
TCN
LSTM/GRU
Transformer
Autoencoder + MLP
```

Mục tiêu:

```text
so sánh physics-based ML với raw/light-curve DL
```

Kỳ vọng:

```text
DL có thể học pattern tốt nhưng dễ overfit
Bayesian Ridge có thể ổn định hơn và dễ giải thích hơn
```

---

### Experiment 6 — Hybrid / ensemble

Kết hợp:

```text
Bayesian Ridge prediction
+ LightGBM residual correction
+ optional CNN prediction
```

Công thức:

$$
\hat{Y}_{\text{final}} = w_1 \hat{Y}_{\text{BR}} + w_2 \hat{Y}_{\text{LGBM}} + w_3 \hat{Y}_{\text{CNN}}
$$

Khuyến nghị:

```text
w_BR lớn nhất, ví dụ 0.6–0.8
w_LGBM 0.1–0.3
w_CNN 0–0.2 nếu CV tốt
```

---

## 9. Final model đề xuất

Mô hình cuối cùng nên chọn:

```text
Bayesian Ridge + Physics-based Features + Target PCA + Sigma Calibration
```

Pipeline final:

```text
1. Load AIRS, FGS, calibration, adc, star_info
2. Apply ADC correction
3. Mask bad/hot pixels
4. Dark subtraction
5. Flat-field correction
6. CDS
7. Temporal binning
8. Extract AIRS wavelength light curves
9. Extract FGS white light curve
10. Normalize and polynomial detrend
11. Detect transit boundaries using FGS
12. Extract depth, shape, noise, multi-scale, stellar features
13. Standardize features
14. Apply PCA to target spectrum
15. Train Bayesian Ridge per PCA component
16. Predict PCA coefficients
17. Inverse PCA to spectrum
18. Estimate sigma from Bayesian std + validation residual
19. Calibrate sigma scale on validation GLL
20. Optional smoothing / clipping
21. Generate submission
```

---

## 10. Pseudocode tổng quát

### 10.1. Feature extraction

```python
for planet_id in train_planets:
    airs = load_airs(planet_id)
    fgs = load_fgs(planet_id)
    calib = load_calibration(planet_id)

    airs = adc_correct(airs, gain, offset)
    fgs = adc_correct(fgs, gain, offset)

    airs = mask_bad_pixels(airs, calib.dead)
    airs = subtract_dark(airs, calib.dark)
    airs = flat_correct(airs, calib.flat)
    airs = cds(airs)
    airs = temporal_bin(airs, n_bins=300)

    fgs = preprocess_fgs(fgs)
    fgs = cds(fgs)
    fgs = temporal_bin(fgs, n_bins=300)

    airs_lc = extract_airs_lightcurves(airs)
    fgs_lc = extract_fgs_lightcurve(fgs)

    transit_bounds = detect_transit(fgs_lc)

    features = []
    features += depth_features(airs_lc, transit_bounds)
    features += multiscale_depth_features(airs_lc, transit_bounds)
    features += shape_features(fgs_lc, transit_bounds)
    features += noise_features(airs_lc, fgs_lc, transit_bounds)
    features += stellar_features(star_info[planet_id])

    X.append(features)

Y = train_targets
```

### 10.2. Training Bayesian Ridge + PCA

```python
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    pca_y = PCA(n_components=30)
    Z_tr = pca_y.fit_transform(Y_tr)

    models = []
    for k in range(Z_tr.shape[1]):
        br = BayesianRidge(max_iter=1000, tol=1e-6)
        br.fit(X_tr, Z_tr[:, k])
        models.append(br)

    Z_pred = []
    Z_std = []

    for br in models:
        mean_k, std_k = br.predict(X_val, return_std=True)
        Z_pred.append(mean_k)
        Z_std.append(std_k)

    Z_pred = np.stack(Z_pred, axis=1)
    Z_std = np.stack(Z_std, axis=1)

    Y_pred = pca_y.inverse_transform(Z_pred)

    Y_var = (Z_std ** 2) @ (pca_y.components_ ** 2)
    Y_std = np.sqrt(Y_var)

    residual_rmse = compute_rmse_per_wavelength(Y_val, Y_pred)
    sigma = np.sqrt(Y_std ** 2 + residual_rmse ** 2)

    sigma = calibrate_sigma(Y_val, Y_pred, sigma)
```

---

## 11. Cách trình bày trong báo cáo

Bố cục báo cáo có thể như sau:

### 11.1. Problem formulation

```text
Input: noisy time-series detector data from AIRS and FGS
Output: atmospheric spectrum + uncertainty
Task: multi-target probabilistic regression
```

### 11.2. Main challenges

```text
weak signal
instrumental noise
systematic effects
high-dimensional time-series
uncertainty calibration
```

### 11.3. Lessons from top solutions

```text
Rank 1: Bayesian physical/sensor model
Rank 2: strong light-curve feature engineering + linear model
Rank 4: limb darkening simulation + linear regression
Rank 5: physical fitting + PCA + Gradient Boosting correction
```

### 11.4. Proposed method

```text
Bayesian Ridge + physics-based features + PCA target + calibrated uncertainty
```

### 11.5. Comparative models

```text
5 pure ML models
5 deep learning models
```

### 11.6. Evaluation

```text
RMSE
Gaussian NLL/GLL
uncertainty calibration plot
residual by wavelength
spectrum smoothness
```

---

## 12. Kết luận

Hướng **Bayesian Ridge + Physics-based Feature Engineering** là lựa chọn hợp lý nhất để làm model mạnh nhất vì nó cân bằng giữa:

- độ chính xác;
- khả năng giải thích;
- khả năng chống overfitting;
- uncertainty estimation;
- sự phù hợp với bản chất vật lý của bài toán transit spectroscopy.

Các model ML thuần như Ridge, Kernel Ridge, ExtraTrees và LightGBM nên được dùng để so sánh. Trong đó, LightGBM có thể đóng vai trò **residual correction** cho Bayesian Ridge.

Các model deep learning như 1D CNN, TCN, LSTM/GRU, Transformer và Autoencoder nên được đưa vào như nhóm baseline học representation tự động, nhưng cần nhấn mạnh rằng chúng dễ overfit hơn nếu số lượng planet train hạn chế.

Final recommendation:

```text
Best model:
Bayesian Ridge + Physics-based Features + Target PCA + Sigma Calibration

Optional improvement:
Bayesian Ridge + LightGBM residual correction + ensemble uncertainty
```
