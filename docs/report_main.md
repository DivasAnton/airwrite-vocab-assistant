# Báo cáo AI - Nhánh `main` (baseline v1)

> Phạm vi: phiên bản baseline tại tag `v1-golden-backup` / commit `38f9789`.
> Số liệu lấy từ `docs/model_card.md`, `artifacts/metadata/baseline_metrics.json` và các manifest của repo.

## 1. Tổng quan hệ thống và thông số triển khai

### Hệ thống làm gì?

Người dùng viết một chữ cái trước webcam. Hệ thống phát hiện bàn tay, lấy tọa độ đầu ngón trỏ,
dựng lại nét trên canvas ảo, chuyển canvas thành ảnh `28x28` rồi CNN dự đoán một trong 26 chữ hoa.

```text
Webcam → MediaPipe Hand Landmarker → Index-finger tracking
        → Gesture/state machine → AirCanvas → preprocessing
        → CNN softmax → label + confidence/top-3
```

### Thông số kỹ thuật chính

| Thành phần | Thông số |
|---|---|
| Camera mặc định | `1280x720`, `30 FPS`, mirror ngang |
| Hand tracking | MediaPipe, 1 bàn tay; detection/presence/tracking threshold mặc định `0.5` |
| Ổn định gesture | 5 frame ổn định; cooldown mặc định `500 ms` |
| Canvas | Nền đen, nét trắng, độ dày mặc định `8 px` |
| Tiền xử lý | grayscale, crop foreground, padding `8 px`, output `28x28`, threshold `20` |
| Model input | `float32`, shape `28x28x1`, range `[0, 1]` |
| Model output | 26 softmax scores theo thứ tự `A-Z` |
| Inference policy v1 | Trả top-1/top-3; baseline cũ chưa bật ngưỡng `UNCERTAIN` ổn định |
| Lưu dữ liệu | Lưu canvas PNG; không lưu raw camera video |
| Runtime | Python 3.11 target, OpenCV, MediaPipe, NumPy, TensorFlow/Keras |

### Ý nghĩa các thông số

- `28x28x1` là hợp đồng bắt buộc giữa preprocessing, lúc train và lúc inference; sai shape hoặc range
  sẽ làm kết quả không còn tương thích.
- `30 FPS` là cấu hình camera mục tiêu, không phải latency inference đã được benchmark.
- Confidence là xác suất softmax, không đồng nghĩa với xác suất đúng đã được calibration.
- Model chỉ nhận **một ký tự hoa mỗi lần**; không nhận raw frame, chữ thường, số hoặc cả từ.

### Giới hạn triển khai

Đây là local prototype chạy trên máy có webcam; repo chưa khai báo ma trận GPU/CPU tối thiểu và chưa có
benchmark latency hoặc soak test camera. Các thông số camera, gesture và preprocessing có thể thay qua
biến môi trường trong `app/utils/config.py`.

## 3. Dataset / Data Collection

### Nguồn dữ liệu

- Dữ liệu tự thu thập từ canvas AirWrite trong workflow Sprint 8, không lưu raw camera frame.
- Nhãn là ký tự tiếng Anh **chữ hoa A-Z** (26 lớp); ảnh đã qua preprocessing của ứng dụng.
- Nguồn nội bộ: `data/raw_airwrite` và `data/manifests/dataset_splits.csv`.
- License: chưa khai báo license riêng cho ảnh tự thu thập; dữ liệu chỉ dùng cho prototype nội bộ.
- Ngày thu thập: metadata mẫu trải từ 2026-07-28 đến trước lần train baseline 2026-08-05.

### Quy mô

| Split | Samples |
|---|---:|
| Scanned | 1,312 |
| Valid unique | 1,306 |
| Training | 914 |
| Validation | 196 |
| Test | 196 |
| Exact duplicates loại | 6 |
| **Total dùng train/evaluate** | **1,306** |

Split stratified, deterministic, seed `42`, tỷ lệ 70/15/15.

### Đặc điểm dữ liệu

- Ngôn ngữ: English alphabet; chỉ chữ hoa.
- Số lớp: 26, phân bố gần đều nhưng số mẫu mỗi lớp còn nhỏ.
- Input: ảnh grayscale `28x28x1`, giá trị float32 sau chuẩn hóa về `[0, 1]`.
- Missing/invalid: mẫu rỗng hoặc không hợp lệ bị loại trong bước scan.
- Duplicate: 6 ảnh trùng chính xác bị loại khỏi manifest.
- Noise/domain variation: khác biệt người viết, độ dày nét, tốc độ viết và camera chưa được bao phủ đầy đủ.

### Data preprocessing

```text
AirWrite canvas snapshot
        ↓
Validate image / reject empty sample
        ↓
Crop foreground + padding, giữ aspect ratio
        ↓
Resize về 28x28, grayscale
        ↓
Normalize uint8 → float32 [0, 1]
        ↓
Exact-duplicate check
        ↓
Stratified train / validation / test split (seed 42)
```

## 4. Exploratory Data Analysis - EDA

- Dataset có đủ 26 lớp trong cả ba split, nhưng quy mô mỗi lớp thấp.
- Tập test chỉ có 196 ảnh, vì vậy một vài dự đoán sai làm thay đổi mạnh recall từng lớp.
- Các dạng dễ nhầm được ghi nhận trong model card: `I/L`, `C/G`, `O/Q`, `F/P`, `U/V`.
- Phân bố chưa đại diện nhiều người viết; chưa có kết luận về khả năng tổng quát ngoài người thu thập.
- Do bài toán classification đa lớp, accuracy nên đi cùng Macro F1, recall từng lớp và confusion matrix.

## 5. Model / Architecture

### Baseline

Logistic Regression trên vector ảnh, dùng làm mốc so sánh.

### Model chính

CNN TensorFlow/Keras:

- Ba convolution block: 32, 64, 128 filters; MaxPooling sau hai block đầu.
- Global Average Pooling → Dense 64 ReLU → Dropout 0.2 → Dense 26 Softmax.
- Augmentation nhẹ khi train: rotation `0.01`, translation `0.03`, zoom `0.03`; không flip.

CNN phù hợp với ảnh nét vẽ vì học được đặc trưng cục bộ và bất biến không gian tốt hơn mô hình tuyến tính.

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Model | `airwrite_character_recognizer` v0.1.0 |
| Framework | TensorFlow/Keras |
| Input | `28x28x1` float32 |
| Classes | 26 (A-Z) |
| Epochs | 50 |
| Batch size | 64 (config mặc định) |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Sparse categorical cross-entropy |
| Early stopping patience | 5 |
| Seed | 42 |

## 7. Training Process

```text
Manifest → TensorFlow dataset → CNN + augmentation → validation
        → checkpoint tốt nhất → test evaluation → export metrics/artifacts
```

Baseline đạt best validation loss ở epoch 50; lịch sử chi tiết nằm tại `artifacts/reports/training_history.csv`.

## 8. Hyperparameter Tuning

Các lần chạy được lưu trong `artifacts/reports/experiments.csv`. Baseline Logistic Regression đạt test
accuracy `68.37%`, Macro F1 `67.05%`; CNN v0.1.0 đạt `79.08%` accuracy và `77.38%` Macro F1.
Chưa có grid search đầy đủ; cấu hình CNN được chọn theo các lần thử augmentation và theo dõi validation.

## 9. Evaluation

| Metric | Score |
|---|---:|
| Validation accuracy | 81.63% |
| Test accuracy | 79.08% |
| Macro precision | 82.06% |
| Macro recall | 79.19% |
| Macro F1 | 77.38% |
| Top-3 accuracy | 90.31% |
| Test samples | 196 |

Top-3 được giữ lại để hỗ trợ correction flow trong runtime.

## 10. Error Analysis

Các lớp có recall test thấp theo model card: `D` 28.6%, `F` 37.5%, `G` 37.5%, `H` 37.5%, `A` 42.9%,
`B` 50.0%. Mẫu lỗi thường liên quan hình dạng tương đồng hoặc nét thiếu/ngắt. Runtime baseline chỉ trả
top class và top-3; chưa có hiệu chỉnh confidence đáng tin cậy. Chi tiết từng lỗi và xác suất nằm trong
`artifacts/reports/error_analysis.csv` và `prediction_probabilities.csv`.

## 11. Comparison

| Model | Accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|
| Majority class | 4.08% | - | - |
| Logistic Regression | 68.37% | 67.05% | 86.73% |
| CNN v0.1.0 | **79.08%** | **77.38%** | **90.31%** |

CNN là lựa chọn hợp lý hơn baseline tuyến tính, nhưng đây vẫn là model chữ hoa đơn ký tự và chưa chứng minh
được khả năng tổng quát đa người viết.
