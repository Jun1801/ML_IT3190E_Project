# Báo cáo tổng hợp — Bài toán, Thách thức cốt lõi & Phương pháp đề xuất

> Tài liệu này tóm tắt phục vụ viết báo cáo: phát biểu bài toán, phân tích **vấn đề lớn nhất**, và **phương pháp đề xuất chủ chốt** cùng lộ trình thực nghiệm. Chi tiết kỹ thuật đầy đủ xem `plans/ariel_2025_model_plan.md`.

---

## 1. Bài toán

**Ariel Data Challenge 2025** (NeurIPS) là bài toán **hồi quy đa mục tiêu có ước lượng bất định** (*multi-target probabilistic regression*): từ dữ liệu quan sát nhiễu của kính thiên văn Ariel khi một ngoại hành tinh đi qua trước sao chủ (transit), dự đoán **phổ truyền qua khí quyển** của hành tinh kèm **độ bất định** tương ứng.

### Đầu vào
| Thành phần | Vai trò |
|---|---|
| AIRS-CH0 signal | Tín hiệu detector theo `[time, spatial, wavelength]` → thông tin phổ |
| FGS1 signal | White-light, SNR cao → bắt timing / shape của transit |
| Calibration (dead, dark, flat) | Hiệu chỉnh detector |
| ADC info (gain/offset) | Hiệu chỉnh raw signal |
| Star info (Rs, Ms, Ts, logg, period) | Thông tin sao chủ |
| train.csv | Ground-truth: phổ 283 bước sóng |

### Đầu ra
Với mỗi hành tinh: **283 giá trị** $\mu_\lambda \approx (R_p/R_s)^2$ (độ sâu transit theo bước sóng) **kèm 283 giá trị bất định** $\sigma_\lambda$.

### Metric chính thức — Gaussian Log-Likelihood (GLL)
Mỗi cặp $(\mu, \sigma)$ được coi là một phân phối Gauss 1 chiều so với ground truth $y$:

$$
\mathrm{GLL}(y, \mu, \sigma) = -\tfrac{1}{2}\Big(\log(2\pi) + \log\sigma^2 + \frac{(y-\mu)^2}{\sigma^2}\Big)
$$

Điểm cuối được **chuẩn hóa về [0, 1]**:

$$
\text{score} = \frac{L_{\text{pred}} - L_{\text{ref}}}{L_{\text{ideal}} - L_{\text{ref}}}, \quad \text{clip}[0,1]
$$

- $L_{\text{ideal}}$: dự đoán hoàn hảo $\mu = y$ với $\sigma = 10$ ppm $= 10^{-5}$.
- $L_{\text{ref}}$: model naive (dự đoán mean & std của tập train cho mọi mẫu).

> **Nguồn:** Kaggle metric *Ariel Gaussian Log Likelihood*; paper *NeurIPS 2024 Ariel Data Challenge* (arXiv 2505.08940).

Metric này được implement chính xác tại `ariel_ml.metrics.ariel_gll_score` (cao hơn = tốt hơn).

---

## 2. Vấn đề lớn nhất của bài toán

Có nhiều khó khăn (tín hiệu yếu, nhiễu instrument, hệ thống systematic, time-series chiều cao). Tuy nhiên **vấn đề cốt lõi quyết định điểm số** là:

> ### ⚑ Ước lượng & hiệu chỉnh độ bất định (uncertainty quantification) dưới metric GLL

Lý do:

1. **Metric chấm $\sigma$ ngang hàng với $\mu$.** GLL chứa cả $\log\sigma^2$ và $(y-\mu)^2/\sigma^2$. Một dự đoán $\mu$ tốt nhưng $\sigma$ sai scale (quá tự tin hoặc quá rộng) bị **phạt rất nặng**, thậm chí về mức naive (score = 0).

2. **Bất định không đồng nhất (heteroscedastic).** Mỗi bước sóng có mức nhiễu khác nhau (SNR phụ thuộc instrument, dead pixel, flat-field), mỗi hành tinh có độ sáng khác nhau. Một $\sigma$ "đúng trung bình" vẫn sai ở phần lớn bước sóng.

3. **Tín hiệu cực yếu.** Độ sâu transit $\sim 10^{-2}$, biến thiên phổ giữa các bước sóng chỉ cỡ $10^{-4}$–$10^{-5}$, lẫn trong nhiễu detector → ranh giới giữa "tín hiệu" và "nhiễu" rất mong manh, khiến calibration $\sigma$ vừa khó vừa quyết định.

4. **Đây là nơi các solution top thắng.** Các lời giải hạng cao đều mạnh ở mô hình sensor/uncertainty, không chỉ ở độ chính xác $\mu$.

**Khoảng trống của baseline thông thường (và của chính pipeline ban đầu):** $\sigma$ chỉ được hiệu chỉnh bằng **một hệ số vô hướng toàn cục** — ép mọi bước sóng và mọi hành tinh dùng chung một scale. Đây chính là dư địa để đề xuất một phương pháp riêng.

---

## 3. Phương pháp đề xuất

### 3.1. Pipeline tổng thể (nền tảng vật lý)

```
Raw AIRS/FGS → detector calibration (ADC, bad-pixel, dark, flat, CDS, binning)
            → light curve extraction (AIRS theo wavelength + FGS white-light)
            → transit boundary detection (dựa FGS, SNR cao)
            → normalize / detrend / smooth
            → physics-based feature engineering (depth, multi-scale, shape, noise, stellar)
            → Target PCA + hồi quy per-component
            → ước lượng σ (Bayesian var + residual RMSE)
            → HIỆU CHỈNH σ (phương pháp đề xuất chủ chốt)
            → phổ + σ → submission
```

### 3.2. So sánh có hệ thống theo **họ mô hình**

Phần comparative study được tổ chức theo 7 họ (đăng ký qua `ModelFactory.families()`), tất cả dùng chung khung **Target PCA + per-component regressor**:

| Họ | Mô hình |
|---|---|
| Linear | Ridge, Lasso, ElasticNet |
| Bayesian / xác suất | **Bayesian Ridge** ⭐, ARD, Gaussian Process, NGBoost |
| Kernel / SVM | SVR, Kernel Ridge |
| Neighbors | KNN |
| Tree ensembles | Random Forest, Extra Trees, Gradient Boosting, HistGB, LightGBM, XGBoost |
| Neural (nông) | MLP |
| Hybrid / ensemble | Residual-corrected, Weighted ensemble |
| Deep (so sánh) | CNN1D, TCN, LSTM, GRU, Transformer, Autoencoder+MLP |

Toàn bộ được so sánh trên **cùng fold** bằng **benchmark harness** (`ariel_ml.benchmark`), xuất bảng RMSE / GLL / coverage / thời gian train, dùng thẳng cho báo cáo.

### 3.3. ⭐ Phương pháp chủ chốt: **PHC — Physics-conditioned Heteroscedastic Calibration**

Ý tưởng: **tách hiệu chỉnh bất định thành một tầng được tối ưu riêng, theo từng bước sóng và bám đặc trưng vật lý, tối ưu trực tiếp trên GLL chính thức.**

#### Bước 1 (đã triển khai) — Hiệu chỉnh σ theo từng bước sóng, có nghiệm đóng tối ưu GLL

Thay hệ số vô hướng $s$ bằng một vector hệ số $s_j$ riêng cho mỗi bước sóng $j$. Vì GLL **cộng tính trên từng phần tử** và $s_j$ **chỉ ảnh hưởng cột $j$**, đạo hàm tổng GLL theo $s_j$ cho **nghiệm đóng**:

$$
\frac{\partial}{\partial s_j}\sum_i \mathrm{GLL}(y_{ij}, \mu_{ij}, s_j\sigma_{ij}) = 0
\;\Longrightarrow\;
\boxed{\,s_j^\star = \sqrt{\tfrac{1}{N}\sum_i \Big(\frac{y_{ij}-\mu_{ij}}{\sigma_{ij}}\Big)^2}\,}
$$

tức **RMS của residual chuẩn hóa theo cột**. Đây đồng thời là scale tối đa hóa GLL — **không cần lặp/tối ưu số**, rất rẻ và có chứng minh chặt.

> Implement: `SigmaCalibrator(per_target=True)`; bật qua `ModelConfig.sigma_per_target` hoặc cờ CLI `--sigma-per-target`. Mặc định vẫn giữ scalar (tương thích ngược).

**Ý nghĩa vật lý:** bước sóng nhiễu cao tự động nhận khoảng tin cậy rộng hơn — đúng bản chất instrument và đúng thứ GLL tưởng thưởng.

#### Bước 2 (đã triển khai) — Scale điều kiện theo đặc trưng nhiễu

$$
\sigma_{ij} = s_j \cdot m_i \cdot \sigma_{ij}, \qquad m_i = \exp\big(\text{features}_i \cdot w\big)
$$

Tương tự $s_j$, **bội số tối ưu GLL cho mỗi hàng cũng có nghiệm đóng**: sau khi áp $s_j$,

$$
t_i = \sqrt{\tfrac{1}{M}\sum_j \Big(\frac{y_{ij}-\mu_{ij}}{s_j\sigma_{ij}}\Big)^2}
$$

Ta **hồi quy $\log t_i$ theo các feature** (ridge log-space) → $g_\theta$ ánh xạ noise features (`oot_std`, `snr_depth`, `cds_noise_proxy`, `bad_pixel_count`…) thành bội số scale. Cho phép bất định thay đổi theo **cả bước sóng lẫn từng hành tinh**.

> Implement: `FeatureConditionedSigmaCalibrator`; bật qua `ModelConfig.sigma_feature_conditioned` hoặc `--sigma-feature-conditioned`.

#### Bước 3 (đã triển khai) — Mixture xác suất các họ mô hình theo GLL

Kết hợp các họ thành một hỗn hợp Gauss, **trọng số = softmax(validation GLL / temperature)**, cộng phương sai đúng kiểu mixture. Biến phần "so sánh các họ" thành **đóng góp ensemble thật sự**.

> Implement: `training.build_gll_weighted_ensemble(...)` → `EnsembleBuildResult` (ensemble + trọng số + điểm GLL từng họ), dùng `WeightedEnsembleRegressor`.

### 3.4. Vì sao phương pháp này độc đáo & hiệu quả

- **Tấn công trực diện scoring rule** thay vì dùng trick chung chung.
- **Có nền lý thuyết** (nghiệm đóng tối ưu GLL per-wavelength) — điểm nhấn học thuật cho báo cáo.
- **Bám vật lý** (calibration theo SNR/nhiễu).
- **Tăng dần được** trên hạ tầng có sẵn, cho **ablation rõ ràng, ra số ngay**.

---

## 4. Kế hoạch thực nghiệm & bảng ablation cho báo cáo

| Thí nghiệm | Nội dung | Metric chính |
|---|---|---|
| Exp 0 | Sanity check dữ liệu, plot light curve / phổ / nhiễu | — |
| Exp 1 | Baseline vật lý đơn giản (depth + Ridge) | RMSE, GLL |
| Exp 2 | So sánh **7 họ mô hình** trên cùng fold (benchmark harness) | RMSE, GLL, coverage |
| Exp 3 | Direct 283-target vs Target PCA | RMSE, GLL |
| **Exp 4** | **Ablation PHC** (xem dưới) | **Ariel GLL** |
| Exp 5 | So sánh nhóm deep sequence (CNN1D/TCN/LSTM/GRU/Transformer/AE) trên light curve thô | RMSE, GLL |
| Exp 6 | Family-mixture theo GLL (PHC bước 3) | Ariel GLL |

**Bảng ablation chủ đạo (Exp 4) — đóng góp cốt lõi:**

| Hiệu chỉnh σ | Ariel GLL |
|---|---|
| Không calibrate | (thấp) |
| Scalar toàn cục (baseline) | baseline |
| **Per-wavelength $s_j$** (PHC bước 1) | + Δ₁ |
| + Feature-conditioned $g_\theta$ (bước 2) | + Δ₂ |
| + Family-mixture theo GLL (bước 3) | + Δ₃ |

> Cả 3 bước PHC đã được triển khai và verify trên dữ liệu synthetic (GLL tăng đơn điệu qua từng bước). Số liệu Δ cụ thể điền sau khi chạy trên dữ liệu Ariel thật.

Kèm **reliability diagram** (coverage thực nghiệm vs danh nghĩa) và biểu đồ $\sigma$ theo bước sóng chồng lên phổ nhiễu để minh họa calibration.

---

## 4b. Thảo luận — phát hiện từ EDA & một ablation âm

**Target gần rank-1 (phát hiện EDA quan trọng).** Phân tích PCA cho thấy **component 1 chiếm ~99.8% variance**: mỗi phổ ≈ một mức nền (độ sâu transit = kích thước planet) gần như phẳng theo bước sóng, còn đặc trưng khí quyển (mục tiêu khoa học) chỉ là modulation ~0.2% chìm trong nhiễu. Điều này giải thích vì sao RMSE gần như không đổi theo `n_components` và vì sao nhiều model có GLL ≈ 0 (đoán đúng mức nền ≈ ngang baseline naive; phần "ăn điểm" là deviation phổ rất khó). Ngoài ra, **kênh AIRS biên (wl-0) nhiễu vượt trội** so với các kênh khác — bằng chứng trực tiếp cho nhu cầu calibration σ **theo từng bước sóng** (PHC).

**Ablation âm — tách mean-depth + shape (`MeanShiftedRegressor`).** Từ phát hiện rank-1, ta thử mô hình hóa tách biệt: một model cho mức nền $d_i=\text{mean}_\lambda y_{i\lambda}$ và một model cho phần dư $r_{i\lambda}=y_{i\lambda}-d_i$ (tín hiệu khí quyển), rồi ghép $\hat y = \hat d + \hat r$. Benchmark trên nhiều cấu hình cho thấy **không cải thiện**: linear (`bayesian_ridge`) chỉ +0.001…+0.009 GLL (không đáng kể), còn tree (`extra_trees`) **tệ hơn** (−0.03…−0.04). Nguyên nhân: **Target-PCA đã ngầm tách rank-1** — component 1 chính là mức nền, các component sau là shape — nên tách tay là dư thừa; với tree, chia thành hai model yếu hơn còn làm giảm chất lượng. Phương pháp được giữ trong code (`MeanShiftedRegressor`, đã test đầy đủ) như một ablation, **không** dùng trong pipeline chính.

---

## 4c. Kết quả thực nghiệm (merged ML + Deep, official Ariel GLL)

**22 model / 7 họ** chấm trên cùng metric Ariel GLL (`merged_benchmark.csv`). Top:

| Hạng | Model | Loại | Ariel GLL | RMSE | σ_mean | cov 1σ |
|---|---|---|---|---|---|---|
| 1 | cnn1d | deep | 0.231 | 0.0028 | — | 0.79 |
| 2 | transformer | deep | 0.226 | 0.0021 | — | 0.72 |
| 3 | extra_trees | ml/trees | 0.215 | 0.0025 | 0.003 | 0.94 |
| 4 | random_forest | ml/trees | 0.199 | 0.0027 | 0.004 | 0.93 |
| 5 | tcn | deep | 0.178 | 0.0031 | — | 0.73 |
| 6 | ngboost | ml/bayesian | 0.167 | 0.0022 | 0.005 | 0.95 |
| 7 | ard | ml/bayesian | 0.143 | 0.0025 | 0.005 | 0.94 |
| 8 | gru | deep | 0.129 | 0.0045 | — | 0.84 |
| 9 | bayesian_ridge | ml/bayesian | 0.093 | 0.0035 | 0.008 | 0.95 |
| 10–22 | autoencoder_mlp, lstm, **+ 11 model GLL = 0** | | ~0 | | | |

**Phát hiện cốt lõi — uncertainty quyết định GLL (2 nhóm tách bạch):**
- **GLL > 0**: model có **σ tự nhiên, chặt, đúng scale** — tree-ensemble variance (extra_trees/random_forest, σ~0.003), Bayesian predictive (ngboost/ard/bayesian_ridge, σ~0.005–0.008), deep log-σ head. Coverage 1σ ≈ 0.72–0.95.
- **GLL = 0** (ridge, lasso, elastic_net, mlp, svr, kernel_ridge, knn, **lightgbm, xgboost, hist_gb**): point-estimator **không có σ gốc** → σ fallback quá rộng (0.28–0.71), coverage = 1.0 → tụt về mức naive.
- Minh chứng đắt: **xgboost RMSE tốt nhất (0.0024) nhưng GLL = 0**. *Mean tốt ≠ điểm GLL cao* — đúng luận điểm PHC: **chất lượng uncertainty mới quyết định**.

### 4c.1. So sánh ĐỒNG BỘ (apples-to-apples) — bảng merged ở trên dễ gây hiểu nhầm

Bảng merged trộn 2 protocol (ML: 3-fold CV + PHC; Deep: single-split, σ thô) nên chỉ tương đối. Khi đánh giá **cùng tập eval + cùng PHC per-wavelength + cùng naive_ref** (`sync_ml_vs_deep.csv`):

| Model | Loại | Ariel GLL (sync) | cov 1σ |
|---|---|---|---|
| **extra_trees** | ml+PHC | **0.287** | 0.77 |
| **random_forest** | ml+PHC | **0.273** | 0.82 |
| cnn1d | deep+PHC | 0.221 | 0.74 |
| transformer | deep+PHC | 0.220 | 0.68 |
| tcn | deep+PHC | 0.167 | 0.74 |
| gru | deep+PHC | 0.119 | 0.81 |
| bayesian_ridge | ml+PHC | 0.074 | 0.67 |
| lstm / autoencoder_mlp | deep+PHC | ~0 | |

→ **Khi so sánh công bằng, tree ensembles THẮNG (0.29/0.27), deep CNN/Transformer bám sát (0.22).** Hiện tượng "deep dẫn đầu" ở bảng merged là **artifact của protocol** (deep single-split không PHC vs ML CV). **Bài học phương pháp luận: phải đồng bộ protocol mới so sánh được.**

### 4c.2. Findings chính (đưa vào report)

1. **Tree ensembles là model tốt nhất** trên GLL chính thức khi so công bằng (extra_trees 0.287, random_forest 0.273); deep cạnh tranh nhưng sau (~0.22). CV xác nhận: extra_trees 0.215 ± 0.025 vs bayesian_ridge 0.093 ± 0.026 (khoảng cách >> std → có ý nghĩa).
2. **Uncertainty quyết định GLL — phân đôi rõ rệt.** Model có σ tự nhiên chặt (trees ~0.003, Bayesian/ngboost/ard ~0.005) → GLL 0.09–0.29; point-estimator (ridge/lasso/elastic_net/mlp/svr/knn/lightgbm/xgboost/hist_gb) → **GLL = 0**, σ 0.28–0.71, coverage = 1.0. **xgboost RMSE tốt nhất (0.0024) nhưng GLL = 0.**
3. **PHC cho lợi ích biên/âm (negative result trung thực).** Ablation calibration (bayesian_ridge): **none 0.127 > scalar 0.093 ≈ per_wavelength 0.093 > feature_conditioned 0.063**. Reliability diagram cho thấy σ của bayesian_ridge **quá rộng (under-confident, over-cover)** → recalibrate càng nới rộng → GLL giảm; `feature_conditioned` overfit. ⇒ **σ nội tại** (Bayesian var + residual-RMSE, hoặc ensemble variance của tree) **đã đủ tốt**; recalibrate tường minh không giúp trên dữ liệu này.
4. **Phụ thuộc dữ liệu mạnh** (learning curve, bayesian_ridge): GLL ≈ 0 khi ≤ 400 planet, **nhảy lên 0.105 tại 800** và 0.093 tại 1100; RMSE giảm đều theo #planet. ⇒ cần ≥ ~800 planet mới vượt baseline naive.
5. **Feature vật lý quan trọng nhất là nhóm NHIỄU + shape FGS**: `fgs_residual_std_after_detrending`, `airs_cds_noise_proxy`, `fgs_oot_std_mean`, `airs_residual_std_after_detrending`, `fgs_oot_std_max`, `fgs_mid_transit_flux`, `fgs_depth_mid_transit`. ⇒ validate feature engineering dựa nhiễu/SNR (không chỉ transit depth).

> **Tổng hợp:** điểm GLL bị chi phối bởi (a) model có uncertainty nội tại tốt và (b) đủ dữ liệu — hơn là bởi tầng recalibrate. Tree ensembles + đủ planet là công thức thắng; PHC nên trình bày trung thực như một hướng calibration có cơ sở lý thuyết nhưng **lợi ích thực nghiệm hạn chế** trên bộ dữ liệu này.

---

## 5. Kết luận cho báo cáo

- **Bài toán:** hồi quy đa mục tiêu có bất định, chấm bằng GLL chuẩn hóa.
- **Vấn đề lớn nhất:** ước lượng & hiệu chỉnh bất định *heteroscedastic* dưới metric GLL — nơi quyết định điểm số và là khoảng trống của baseline.
- **Phương pháp đề xuất:** pipeline vật lý + Target PCA + benchmark đa-họ trên metric Ariel chính thức, kèm PHC (hiệu chỉnh $\sigma$ per-wavelength có nghiệm đóng tối ưu GLL).
- **Kết quả thực nghiệm (trung thực):**
  - **Model tốt nhất = tree ensembles** (extra_trees/random_forest, GLL ~0.27–0.29 khi so công bằng); deep CNN/Transformer cạnh tranh (~0.22). Bayesian linear ~0.09.
  - **Phát hiện chính: chất lượng uncertainty nội tại + đủ dữ liệu quyết định GLL** — point-estimator (mean tốt, kể cả xgboost) đều GLL = 0 vì σ không tin cậy.
  - **PHC: kết quả âm trung thực** — recalibrate tường minh cho lợi ích biên/âm vì σ nội tại đã đủ tốt (thậm chí hơi rộng); `feature_conditioned` overfit. PHC vẫn có giá trị lý thuyết (nghiệm đóng) và để chuẩn hóa σ giữa các model khi so sánh, nhưng **không phải đòn bẩy chính** trên dữ liệu này.
- **Bài học phương pháp luận:** phải đồng bộ protocol (cùng split + cùng calibration) mới so sánh ML vs Deep công bằng — nếu không sẽ kết luận sai ("deep thắng" là artifact).
