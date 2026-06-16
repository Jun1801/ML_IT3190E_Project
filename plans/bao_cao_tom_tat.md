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

#### Bước 2 (lộ trình) — Scale điều kiện theo đặc trưng nhiễu

$$
\sigma_j(x) = s_j \cdot g_\theta\big(\text{noise\_features}(x)\big)
$$

với $g_\theta$ ánh xạ các feature nhiễu sẵn có (`oot_std`, `snr_depth`, `cds_noise_proxy`, `bad_pixel_count`…) → bội số scale theo từng quan sát. Cho phép bất định thay đổi theo **cả bước sóng lẫn từng hành tinh**.

#### Bước 3 (lộ trình) — Mixture xác suất các họ mô hình theo GLL

Kết hợp các họ thành một hỗn hợp Gauss, **trọng số = softmax(validation GLL)**, cộng phương sai đúng kiểu mixture (đã có sẵn trong `WeightedEnsembleRegressor`). Biến phần "so sánh các họ" thành **đóng góp ensemble thật sự**.

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
| Exp 5 | So sánh nhóm deep learning | RMSE, GLL |
| Exp 6 | Family-mixture theo GLL (PHC bước 3) | Ariel GLL |

**Bảng ablation chủ đạo (Exp 4) — đóng góp cốt lõi:**

| Hiệu chỉnh σ | Ariel GLL |
|---|---|
| Không calibrate | (thấp) |
| Scalar toàn cục (baseline) | baseline |
| **Per-wavelength $s_j$** (PHC bước 1 — đã làm) | + Δ₁ |
| + Feature-conditioned $g_\theta$ (bước 2) | + Δ₂ |
| + Family-mixture theo GLL (bước 3) | + Δ₃ |

Kèm **reliability diagram** (coverage thực nghiệm vs danh nghĩa) và biểu đồ $\sigma$ theo bước sóng chồng lên phổ nhiễu để minh họa calibration.

---

## 5. Kết luận cho báo cáo

- **Bài toán:** hồi quy đa mục tiêu có bất định, chấm bằng GLL chuẩn hóa.
- **Vấn đề lớn nhất:** ước lượng & hiệu chỉnh bất định *heteroscedastic* dưới metric GLL — nơi quyết định điểm số và là khoảng trống của baseline.
- **Phương pháp đề xuất:** pipeline vật lý + Bayesian Ridge + Target PCA, với **đóng góp chủ chốt là PHC** — hiệu chỉnh $\sigma$ theo từng bước sóng (nghiệm đóng tối ưu GLL), mở rộng sang điều kiện-theo-feature và mixture các họ mô hình. Tất cả được đánh giá thống nhất bằng metric Ariel chính thức trên một benchmark harness chung.
