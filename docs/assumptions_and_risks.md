# Assumptions and Risks

## 1. Mục tiêu

Tài liệu này liệt kê:

- Các giả định chưa được chứng minh.
- Rủi ro kỹ thuật.
- Rủi ro sản phẩm.
- Rủi ro privacy.
- Cách kiểm chứng.
- Hướng giảm thiểu.

## 2. Product Assumptions

### Assumption 1

Người dùng thấy air-writing tiện hơn chuyển tab và gõ từ.

Cách kiểm chứng:

- Đo thời gian hoàn thành cùng một từ.
- Phỏng vấn người dùng.
- Quan sát mức độ gián đoạn.
- So sánh với keyboard input.

### Assumption 2

Người dùng có thể viết ký tự đủ rõ để model nhận dạng.

Cách kiểm chứng:

- Thu thập canvas thực tế.
- Đo accuracy theo người dùng.
- Kiểm tra nhiều kiểu viết.

### Assumption 3

Nhận dạng từng ký tự không làm flow quá chậm.

Cách kiểm chứng:

- Đo thời gian cho từ 4, 6 và 10 ký tự.
- Hỏi người dùng về mức mệt.
- Đo retry rate.

### Assumption 4

Correction flow giúp người dùng chấp nhận model chưa hoàn hảo.

Cách kiểm chứng:

- Đo correction completion rate.
- Đo thời gian sửa.
- Đo tỷ lệ bỏ flow.

### Assumption 5

Vocabulary history và review tạo retention.

Cách kiểm chứng:

- Đo save rate.
- Đo tỷ lệ quay lại.
- Đo số từ được review.

### Assumption 6

Người dùng chấp nhận bật camera nếu xử lý local.

Cách kiểm chứng:

- Phỏng vấn.
- Privacy onboarding.
- Theo dõi tỷ lệ từ chối camera.

## 3. Risk Register

| Rủi ro | Loại | Mức độ | Tác động | Cách kiểm chứng | Hướng giảm thiểu |
|---|---|---:|---|---|---|
| Air-writing chậm hơn gõ | Product | Rất cao | Sản phẩm không có giá trị | Đo thời gian | Tối giản gesture, extension overlay |
| Người dùng mỏi tay | UX | Cao | Không muốn dùng lại | Test session dài | Giảm thời gian viết, hỗ trợ shortcut |
| Viết từng ký tự quá chậm | UX | Rất cao | Flow bị bỏ | Đo với nhiều độ dài từ | Sau MVP nghiên cứu word recognition |
| MediaPipe rung | CV | Cao | Nét méo | Test camera thực tế | Smoothing, threshold movement |
| Mất landmark | CV | Cao | Stroke đứt | Test nhiều điều kiện | State recovery, interpolation |
| Gesture kích hoạt nhầm | CV/UX | Cao | Clear hoặc stop sai | Test gesture | Giảm số gesture, dùng keyboard fallback |
| Dataset mismatch | ML | Rất cao | Accuracy thực tế thấp | Air-writing test set | Thu thập dữ liệu thật, fine-tune |
| Confidence không đáng tin | ML | Cao | Gợi ý sai | Calibration test | Threshold, correction warning |
| Preprocessing cắt mất nét | ML | Cao | Prediction sai | Unit test và visual test | Padding, bounding box validation |
| Camera permission bị từ chối | Privacy | Cao | Không dùng được | User test | Privacy notice, local processing |
| Lưu raw video ngoài ý muốn | Privacy | Rất cao | Vi phạm niềm tin | Code review | Không implement recording |
| Gửi video lên backend | Privacy | Rất cao | Rủi ro dữ liệu | Architecture review | Chỉ gửi canvas |
| Dictionary API lỗi | Backend | Trung bình | Không có nghĩa | Failure test | Cache, local fallback |
| Scope quá lớn | Project | Rất cao | Không hoàn thành MVP | Sprint review | Khóa MVP scope |
| Chuyển web quá sớm | Architecture | Cao | Rework lớn | Exit criteria | Hoàn thành local prototype trước |
| Tất cả logic trong main.py | Architecture | Cao | Khó bảo trì | Code review | Module boundaries |
| Không có correction flow | Product | Rất cao | Model sai làm flow thất bại | UX review | Correction bắt buộc |
| Lưu duplicate sai logic | Data | Trung bình | History bẩn | Unit test | Unique rule và upsert |
| Hardcode config | Architecture | Trung bình | Khó deploy | Code review | Config module và env |
| Không có test data thật | ML | Rất cao | Đánh giá sai | Dataset audit | Tạo real-world test set |

## 4. Computer Vision Risks

### Jitter

Nguyên nhân:

- Camera noise.
- Landmark prediction không ổn định.
- Tay di chuyển nhỏ.
- FPS không đều.

Giảm thiểu:

- Moving average.
- Exponential smoothing.
- Bỏ qua chuyển động dưới threshold.
- Giới hạn khoảng nhảy tối đa.

### Stroke Disconnection

Nguyên nhân:

- Mất hand detection.
- Previous point không được reset đúng.
- FPS thấp.

Giảm thiểu:

- State recovery.
- Reset previous point khi mất tay.
- Không nối các điểm quá xa.

### Wrong Gesture Detection

Nguyên nhân:

- Gesture rule đơn giản quá.
- Góc camera thay đổi.
- Người dùng xoay tay.

Giảm thiểu:

- Gesture tối thiểu.
- Debounce theo số frame.
- Keyboard fallback ở prototype.

## 5. Machine Learning Risks

### Dataset Mismatch

Dataset chữ viết tay truyền thống khác air-writing canvas ở:

- Nét vẽ.
- Độ dày.
- Kích thước.
- Tỷ lệ.
- Điểm bắt đầu và kết thúc.
- Không có texture giấy.
- Smoothing.

Giảm thiểu:

- Thu thập dataset air-writing.
- Augmentation phù hợp.
- Chuẩn hóa preprocessing.
- Đánh giá riêng real-world set.

### Overfitting

Giảm thiểu:

- Train/validation/test split.
- Augmentation.
- Early stopping.
- Regularization.
- Theo dõi validation loss.

### Incorrect Label Mapping

Giảm thiểu:

- Lưu mapping cùng model.
- Không hardcode nhiều nơi.
- Test mapping.

## 6. Product Risks

### Novelty Without Utility

Người dùng có thể thấy thú vị nhưng không dùng lại.

Kiểm chứng:

- Return intent.
- Repeat usage.
- So sánh với keyboard.

### Too Many Steps

Nếu mỗi ký tự cần quá nhiều gesture, flow trở nên khó chịu.

Giảm thiểu:

- Tối giản gesture.
- Auto-clear sau confirm.
- Keyboard shortcut.
- Gợi ý sửa nhanh.

### Poor Correction UX

Nếu sửa còn chậm hơn viết lại, sản phẩm thất bại.

Giảm thiểu:

- Suggestion list.
- Inline editing.
- Backspace character.
- Confirm nhanh.

## 7. Privacy Risks

### Camera Anxiety

Giảm thiểu:

- Camera indicator.
- Privacy notice.
- Local processing.
- Không recording.
- Tắt camera dễ dàng.

### Unnecessary Data Retention

Giảm thiểu:

- Không lưu raw frame.
- Chỉ lưu canvas khi user đồng ý.
- Có retention rule.
- Có delete function.

## 8. Validation Plan

### Validation A — Tracking

Câu hỏi:

> Landmark có đủ ổn để tạo nét chữ rõ không?

Thực hiện ở Sprint 3–5.

### Validation B — Canvas

Câu hỏi:

> Canvas có đủ sạch và nhất quán để dùng cho model không?

Thực hiện ở Sprint 7–8.

### Validation C — Recognition

Câu hỏi:

> Model có nhận đúng air-writing canvas thực tế không?

Thực hiện ở Sprint 9–10.

### Validation D — Character-by-character UX

Câu hỏi:

> Người dùng có chấp nhận viết từng ký tự không?

Thực hiện ở Sprint 11–15.

### Validation E — Product Value

Câu hỏi:

> AirWrite có giảm gián đoạn khi học qua video không?

Thực hiện sau local prototype và web MVP.

## 9. Risk Review Rule

Cuối mỗi Sprint, cần hỏi:

- Có rủi ro mới không?
- Rủi ro nào đã giảm?
- Rủi ro nào tăng?
- Có giả định nào đã bị bác bỏ?
- Có cần thay đổi scope không?
- Có dữ liệu nào chứng minh quyết định hiện tại không?
