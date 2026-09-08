# Báo cáo AI - Nhánh `v2` (EMNIST identity + whole-word)

> Phạm vi: pipeline hiện tại tại commit `0d33175`. Hai ref `main` và `airwrite-v2` hiện cùng trỏ commit này;
> báo cáo này mô tả phần nâng cấp v2 (Sprint 9E-11W), còn baseline lịch sử được tách ở `report_main.md`.

## 1. Tổng quan hệ thống và thông số triển khai

### Hệ thống làm gì?

V2 dùng cùng pipeline webcam/canvas của ứng dụng nhưng thay model chữ hoa custom bằng model EMNIST
letter-identity. Model dự đoán identity `a-z`; case được quyết định bởi chế độ người dùng. Ở Word Mode,
các segment chữ rời được gom thành draft để review trước khi commit toàn bộ từ.

```text
Webcam → Hand landmarks → Finger/gesture state machine → AirCanvas
        → crop/segment → shared 28x28 contract → EMNIST CNN E02
        → identity Top-K → case resolver → Word Builder / Whole-word draft
```

### Thông số kỹ thuật chính

| Thành phần | Thông số |
|---|---|
| Camera mặc định | `1280x720`, `30 FPS`, mirror ngang |
| Hand tracking | MediaPipe, 1 bàn tay; detection/presence/tracking threshold `0.5` |
| Canvas và preprocessing | Nền đen/nét trắng; crop foreground; padding `8 px`; output `28x28`; threshold `20` |
| Model input | normalized `float32`, shape `28x28x1`, foreground sáng trên nền đen |
| Model output | 26 identity softmax theo canonical order `a-z` |
| Case handling | `LOWERCASE`, `UPPERCASE`, `CAPITALIZE_FIRST`, `CUSTOM`; model không tự đoán case |
| Prediction acceptance | Top-1 confidence `>=0.60` và margin Top-1/Top-2 `>=0.15` |
| Candidate correction | Top-3; chọn bằng `1/2/3`, hủy bằng `X` |
| Character Mode | Một chữ cho mỗi lần `DONE` |
| Word Mode | Segment chữ rời, review draft, tách/gộp, accept/cancel, commit nguyên tử |
| Runtime | Python 3.11 target, OpenCV, MediaPipe, NumPy, TensorFlow/Keras |
| Privacy | Chỉ lưu canvas/dataset image local; không lưu raw camera video |

### Thông số model và hệ thống cần hiểu đúng

- E02 là **identity model**, nên `a` và `A` dùng cùng class; `CaseLabelResolver` chỉ đổi cách render.
- `Top-1 confidence` và `Top-1 - Top-2 margin` là hai điều kiện độc lập để tránh chấp nhận dự đoán mơ hồ.
- Segment confidence (tách chữ trong Word Mode) khác model confidence; một segment tách sai có thể làm
  cả chuỗi từ sai dù CNN nhận dạng tốt.
- `DONE` chỉ tạo một prediction event; kết quả `UNCERTAIN` được giữ pending để người dùng sửa trước khi commit.

### Giới hạn triển khai

V2 vẫn là local prototype. Repo chưa có benchmark latency GPU/CPU, ma trận phần cứng tối thiểu hay manual
webcam audit hoàn chỉnh cho các từ mục tiêu. Các giá trị camera, threshold, model path và whole-word policy
có thể cấu hình qua `app/utils/config.py` và biến môi trường.

## 3. Dataset / Data Collection

### Nguồn dữ liệu

- **EMNIST Letters chính thức** (NIST, phân phối IDX/GZIP): nguồn train/test chuẩn cho model identity.
- **AirWrite custom**: ảnh canvas tự thu thập để kiểm tra domain gap, lưu local tại `data/raw_airwrite`.
- EMNIST gộp dạng hoa/thường thành 26 identity `a-z`; case không phải nhãn của model.
- License/điều khoản: EMNIST sử dụng theo điều khoản của nguồn NIST/EMNIST; cần giữ attribution khi phát hành.
- Ngày audit/đánh giá trong repo: 2026-08 đến 2026-09; seed split `42`.

### Quy mô

| Dataset / Split | Samples |
|---|---:|
| EMNIST official train | 124,800 |
| EMNIST derived train | 112,320 |
| EMNIST validation | 12,480 |
| EMNIST official test | 20,800 |
| AirWrite scanned | 3,842 |
| AirWrite valid unique | 3,835 |
| AirWrite train / validation / test | 2,685 / 575 / 575 |

### Đặc điểm dữ liệu

- 26 lớp identity, canonical order `a-z`, raw label EMNIST `1-26` map về `0-25`.
- Ảnh grayscale `28x28`; foreground sáng trên nền đen sau transpose và normalize.
- AirWrite custom có 3,835 mẫu hợp lệ; lớp nhỏ nhất `O=99`, lớn nhất `A/B/C/D/F/G/I/J/L/M/N/P/R/S/T/W/X/Y/Z=150`.
- Imbalance ratio `1.515`; 5 exact duplicates và 2 lỗi preprocessing được ghi nhận.
- Chưa có số đo độ dài câu vì task là nhận dạng ký tự; whole-word được tạo từ các segment ký tự rời.

### Data preprocessing

```text
EMNIST IDX / AirWrite canvas
        ↓
Read + validate shape/labels
        ↓
EMNIST: transpose đúng một lần; AirWrite: crop foreground + giữ aspect ratio
        ↓
Shared 28x28 grayscale contract
        ↓
Normalize uint8 → float32 / 255
        ↓
Duplicate + validation audit (AirWrite)
        ↓
EMNIST stratified train/validation; official test giữ nguyên
```

## 4. Exploratory Data Analysis - EDA

- EMNIST cân bằng tuyệt đối theo identity: 4,800 train và 800 test mỗi lớp.
- AirWrite lệch vừa phải, nổi bật lớp `O` ít mẫu hơn; cần bổ sung phiên thu thập độc lập.
- Official EMNIST và AirWrite có khác biệt lớn về độ dày, kích thước và hình học nét, tạo domain gap.
- AirWrite được đánh giá riêng theo style: uppercase và lowercase; không trộn kết quả với official EMNIST.
- Accuracy vẫn chưa đủ: báo cáo dùng Macro F1, Top-3 và confusion/error analysis.

## 5. Model / Architecture

### Baseline

E01: cùng CNN nhưng tắt augmentation, dùng để so sánh trên cùng validation split.

### Model chính - E02

CNN TensorFlow/Keras gồm ba block convolution 32/64/128 filters, Batch Normalization sau mỗi convolution,
MaxPooling sau hai block đầu, Global Average Pooling, Dense 128 ReLU, Dropout 0.30 và Dense 26 Softmax.
E02 bật light rotation/translation/zoom. Model chỉ dự đoán identity, còn `CaseLabelResolver` dùng case mode
do người dùng chọn để render hoa/thường.

Lựa chọn này giữ được đặc trưng ảnh nét vẽ, tận dụng dữ liệu EMNIST lớn và tách rõ identity khỏi case control.

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Model | `airwrite_emnist_letter_identity` E02 v1.0.0 |
| Framework | TensorFlow/Keras |
| Input | `28x28x1` float32 |
| Classes | 26 (`a-z` identity) |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Dropout | 0.30 |
| Augmentation | rotation 0.03, translation 0.08, zoom 0.08 |
| Max epochs / early stopping | 50 / 5 |
| Random seed | 42 |

## 7. Training Process

```text
Official EMNIST → persisted stratified split → E01/E02 train
        ↓
Compare validation accuracy/F1 on same split
        ↓
Select E02 → evaluate once on official test
        ↓
Evaluate independent AirWrite custom set by style
        ↓
Export model bundle, labels, preprocessing contract and reports
```

E02 đạt validation accuracy `94.89%`, validation Macro F1 `94.89%`; official test chỉ được mở sau khi chọn
experiment. Training history và metadata nằm trong các artifact của experiment E02.

## 8. Hyperparameter Tuning

| Experiment | Augmentation | Validation accuracy | Validation Macro F1 |
|---|---|---:|---:|
| E01 | Off | baseline để đối chiếu | baseline để đối chiếu |
| E02 | Light | **94.89%** | **94.89%** |

E02 được chọn vì augmentation nhẹ cải thiện khả năng chịu lệch/vị trí nét trên validation. Official test
không được dùng để chọn hyperparameter.

## 9. Evaluation

| Evaluation set | Accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|
| EMNIST official test (20,800) | 94.37% | 94.38% | 99.50% |
| AirWrite custom (1,600 unique) | 64.12% | 64.39% | 81.50% |
| AirWrite uppercase style (1,306) | 62.17% | - | - |
| AirWrite lowercase style (294) | 72.79% | - | - |

Metric chính là accuracy/Macro F1 cho classification; Top-3 phản ánh khả năng hỗ trợ người dùng sửa candidate.

## 10. Error Analysis

Domain gap là lỗi lớn nhất: model rất tốt trên EMNIST nhưng giảm còn 64.12% trên AirWrite. Các identity yếu
trên AirWrite gồm `r`, `n`, `b`, `d`, `e`, `x`, `u`; nguyên nhân khả dĩ là hình dạng người viết, stroke
thickness và content size khác dữ liệu EMNIST. Runtime v2 giữ Top-3, trạng thái `UNCERTAIN` và cho phép chọn
candidate `1/2/3`; whole-word cho phép review, tách/gộp segment và commit nguyên tử. Model không tự suy luận case.

## 11. Comparison

| Model / domain | Accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|
| v1 CNN custom AirWrite | 79.08% (test 196) | 77.38% | 90.31% |
| v2 E02 trên EMNIST | **94.37%** | **94.38%** | **99.50%** |
| v2 E02 trên AirWrite | 64.12% | 64.39% | 81.50% |

So sánh cho thấy v2 học identity tốt trên nguồn dữ liệu lớn và có workflow whole-word/case-preserving tốt hơn,
nhưng chưa vượt baseline v1 trên cùng domain AirWrite uppercase. Ưu tiên tiếp theo là thu thập nhiều người viết,
đồng nhất preprocessing và đánh giá webcam thật trước khi kết luận production readiness.
