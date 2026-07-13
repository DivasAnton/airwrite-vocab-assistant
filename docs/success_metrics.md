# Success Metrics

## 1. Mục tiêu

Tài liệu này xác định cách đánh giá:

- Computer Vision.
- Machine Learning.
- Product UX.
- Privacy.
- Reliability.
- Business potential.

Không được kết luận project “thành công” chỉ vì camera hoặc model chạy được.

## 2. Computer Vision Metrics

### Camera Stability

Mục tiêu:

- Camera mở thành công.
- Chạy liên tục ít nhất 10 phút trong điều kiện bình thường.
- Không memory leak rõ ràng.
- Không crash khi mất camera tạm thời.

### Hand Detection Rate

Đo:

```text
Số frame detect được tay / tổng số frame có tay trong vùng camera
```

Mục tiêu ban đầu:

- Đủ ổn trong ánh sáng bình thường.
- Không cần tối ưu cho mọi điều kiện ngay từ MVP.

### Finger Tracking Stability

Theo dõi:

- Mức rung tọa độ.
- Số lần landmark bị mất.
- Khoảng cách nhảy bất thường giữa hai frame.
- Tỷ lệ stroke bị đứt.

### Drawing Latency

Mục tiêu:

- Nét vẽ phản hồi gần thời gian thực.
- Người dùng không cảm nhận độ trễ lớn giữa tay và canvas.

### State Accuracy

Kiểm tra:

- Không vẽ khi IDLE.
- Không vẽ khi PAUSED.
- Chỉ vẽ khi WRITING.
- DONE chỉ xảy ra khi người dùng kết thúc.
- CLEAR không kích hoạt ngoài ý muốn.

## 3. Preprocessing Metrics

### Crop Quality

- Không cắt mất nét.
- Không giữ quá nhiều khoảng trắng.
- Bounding box đúng vùng chữ.

### Output Shape

- Đúng shape model yêu cầu.
- Đúng số channel.
- Đúng dtype.
- Đúng range normalize.

### Empty Canvas Handling

- Không crash.
- Không gọi model.
- Trả lỗi rõ ràng.

### Train/Inference Consistency

- Cùng resize rule.
- Cùng normalization.
- Cùng foreground/background.
- Cùng channel format.
- Cùng label mapping.

## 4. Machine Learning Metrics

### Dataset Accuracy

Mục tiêu ban đầu:

- Test accuracy tham khảo từ 90% trở lên.

Lưu ý:

- Đây không phải tiêu chí duy nhất.
- Accuracy cao trên dataset không đảm bảo tốt với air-writing canvas.

### Real Air-writing Accuracy

Phải tạo tập test riêng từ:

- Chính người phát triển.
- Nhiều lần viết.
- Nhiều kích thước.
- Nhiều tốc độ.
- Nhiều điều kiện ánh sáng.
- Nhiều người dùng nếu có thể.

Metric:

```text
Số ký tự nhận đúng / tổng số ký tự thực tế
```

### Per-class Accuracy

Theo dõi từng ký tự:

- Ký tự nào dễ nhận.
- Ký tự nào dễ nhầm.
- Ký tự nào thiếu dữ liệu.

### Confusion Matrix

Dùng để tìm cặp dễ nhầm:

- o và 0.
- c và e.
- l và i.
- u và v.
- m và n.
- p và q.
- b và d.

### Confidence Quality

Không chỉ trả confidence, cần kiểm tra:

- Confidence cao có tương ứng với dự đoán đúng không?
- Prediction sai có confidence quá cao không?
- Threshold nào nên kích hoạt correction warning?

### Inference Latency

Đo thời gian từ:

```text
Preprocessed image
→ model inference
→ label + confidence
```

Mục tiêu:

- Đủ nhanh để người dùng không thấy chờ lâu cho từng ký tự.

## 5. Product UX Metrics

### Task Completion Rate

```text
Số người hoàn thành flow / số người bắt đầu flow
```

Flow:

```text
Write → Recognize → Correct → Translate → Save
```

### Time to Lookup

Đo từ lúc người dùng bắt đầu viết đến khi có nghĩa.

Cần so sánh với:

- Gõ bàn phím.
- Chuyển tab.
- Voice input nếu có.

### Correction Rate

```text
Số prediction cần sửa / tổng số prediction
```

Correction rate cao cho thấy:

- Model chưa đủ tốt.
- Preprocessing sai.
- UX viết khó.
- Dataset mismatch.

### Retry Rate

```text
Số lần phải viết lại / tổng số ký tự
```

### Save Rate

```text
Số từ được lưu / số từ được tra
```

### Return Rate

Sau khi có web MVP:

- Người dùng quay lại trong 1 ngày.
- Người dùng quay lại trong 7 ngày.
- Người dùng quay lại để ôn từ.

### User Understanding

Người dùng phải hiểu:

- Trạng thái hiện tại.
- Khi nào bắt đầu viết.
- Khi nào kết thúc.
- Cách clear.
- Cách sửa.
- Cách lưu.

## 6. Privacy Metrics

- Không lưu raw video.
- Không gửi video stream.
- Camera indicator rõ.
- Có nút tắt camera.
- Dữ liệu lưu có mục đích rõ.
- Có chức năng xóa dữ liệu.
- Không log thông tin nhạy cảm.
- Không lưu ảnh canvas ngoài ý muốn.

## 7. Reliability Metrics

Theo dõi:

- Camera initialization failure.
- Hand detection failure.
- Empty canvas error.
- Model load failure.
- Dictionary lookup failure.
- Save failure.
- Corrupted local storage.
- Unexpected exception.

## 8. Sprint-Level Acceptance Metrics

### Sprint 2–5

- Camera mở.
- Detect tay.
- Track ngón trỏ.
- Vẽ được nét.

### Sprint 6–8

- Gesture ổn.
- Save image.
- Preprocessing đúng.

### Sprint 9–10

- Model train.
- Model load.
- Predict label + confidence.

### Sprint 11–15

- Ghép từ.
- Correction.
- Translation.
- Local storage.
- UX polish.

## 9. MVP Success Criteria

MVP được coi là thành công kỹ thuật khi:

- Core flow chạy end-to-end.
- Không cần sửa code giữa flow.
- Có error handling cơ bản.
- Có correction flow.
- Có local storage.
- Có test plan.
- Không vi phạm privacy principles.

MVP được coi là có tín hiệu sản phẩm khi:

- Người dùng hoàn thành flow.
- Người dùng thấy flow có giá trị.
- Thời gian không quá chậm.
- Correction không quá phiền.
- Người dùng lưu từ.
- Người dùng muốn sử dụng lại.

## 10. Anti-Metrics

Không dùng các chỉ số sau làm bằng chứng duy nhất:

- Camera mở được.
- MediaPipe detect được tay.
- Model đạt accuracy cao trên dataset.
- Demo chạy một lần.
- Có nhiều dòng code.
- Có nhiều tính năng.
- UI đẹp.
- Có Docker.
- Có deploy.

Các chỉ số này có thể hữu ích nhưng không chứng minh sản phẩm giải quyết đúng vấn đề.
