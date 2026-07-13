# MVP Scope

## 1. Mục tiêu MVP

MVP đầu tiên phải kiểm chứng được rằng người dùng có thể:

1. Viết từng ký tự trong không khí.
2. Nhìn thấy nét viết trên canvas.
3. Nhận kết quả dự đoán.
4. Sửa kết quả nếu sai.
5. Ghép ký tự thành từ.
6. Tra nghĩa.
7. Xem ví dụ.
8. Lưu từ.
9. Xem lại lịch sử từ.

MVP không nhằm chứng minh rằng toàn bộ sản phẩm đã sẵn sàng thương mại hóa.

## 2. Core Flow

```text
Write → Recognize → Correct → Translate → Save → Review
```

Mỗi phần của MVP phải phục vụ trực tiếp cho flow này.

## 3. Must Have

### Camera

- Mở webcam.
- Kiểm tra camera không mở được.
- Hiển thị frame.
- Flip hình theo chế độ mirror nếu cần.
- Dừng camera an toàn.

### Hand Detection

- Detect ít nhất một bàn tay.
- Lấy landmarks.
- Xử lý trường hợp không có tay.
- Không crash khi MediaPipe không trả kết quả.

### Finger Tracking

- Lấy landmark đầu ngón trỏ.
- Convert normalized coordinate sang pixel.
- Theo dõi current point.
- Quản lý previous point.
- Có smoothing cơ bản.

### Gesture Control

- Có trạng thái IDLE.
- Có trạng thái READY.
- Có trạng thái WRITING.
- Có trạng thái DONE.
- Có CLEAR.
- Không vẽ ngoài WRITING.

### Air Drawing Canvas

- Vẽ line giữa previous point và current point.
- Clear canvas.
- Hiển thị canvas.
- Kiểm tra canvas rỗng.
- Reset stroke đúng cách.

### Image Saving

- Lưu ảnh ký tự.
- Không lưu raw camera frame.
- Tạo tên file an toàn.
- Tách thư mục output.

### Image Preprocessing

- Crop bounding box.
- Add padding.
- Resize đúng kích thước model.
- Normalize.
- Xử lý foreground/background.
- Xử lý canvas rỗng.
- Giữ đúng aspect ratio nếu cần.

### Character Recognition

- Load model.
- Predict label.
- Trả confidence.
- Có label mapping.
- Xử lý input sai shape.
- Không hardcode model path trong nhiều file.

### Word Builder

- Append ký tự.
- Xóa ký tự cuối.
- Clear từ.
- Confirm từ.
- Normalize output.

### Correction Flow

- Hiển thị predicted text.
- Hiển thị confidence.
- Cho phép sửa thủ công.
- Có thể gợi ý từ gần đúng.
- Chỉ lưu corrected text.

### Translation

- Tra nghĩa tiếng Việt.
- Từ loại.
- Phiên âm nếu có.
- Ví dụ tiếng Anh.
- Dịch ví dụ sang tiếng Việt.
- Xử lý từ không tìm thấy.

### Local Vocabulary Storage

- Lưu từ.
- Không tạo duplicate sai logic.
- Xem lịch sử.
- Lưu ngày tạo.
- Lưu source.
- Có thể xóa từ.

## 4. Should Have

- Smoothing tọa độ.
- FPS display cho debug.
- Logging.
- Config tập trung.
- Unit test preprocessing.
- Unit test canvas.
- Unit test word builder.
- Test predictor bằng ảnh tĩnh.
- Spell suggestion.
- Favorite word.
- Search vocabulary.
- Hiển thị trạng thái rõ.
- Retry camera.
- Model version trong recognition log.

## 5. Could Have

- Phiên âm audio.
- Export CSV.
- Flashcard cơ bản.
- Thống kê số từ.
- Difficulty level.
- Favorite.
- Tag từ.
- Source video.
- Local usage analytics.
- Theme tối.
- Keyboard shortcut.
- Simple onboarding.

## 6. Won’t Have Now

- Nhận dạng nguyên từ trong một lần viết.
- Chrome Extension.
- PWA.
- Mobile App.
- Payment.
- Subscription.
- Multi-language.
- Teacher dashboard.
- Class management.
- Social login.
- Cloud sync.
- AI tutor.
- Live collaborative learning.
- OCR phụ đề tự động.
- Voice input.
- Fine-tune Transformer lớn.
- B2B features.
- Advanced analytics.

## 7. Non-functional Requirements

### Correctness

- Không predict canvas rỗng.
- Không lưu từ chưa xác nhận.
- Train preprocessing và inference preprocessing phải thống nhất.
- Confidence phải được trả về.
- Không tạo nghĩa giả khi lookup thất bại.

### Privacy

- Không lưu raw video.
- Không gửi video lên server.
- Chỉ lưu canvas khi cần.
- Có thông báo camera.
- Có thể xóa dữ liệu.

### Security

- Không hardcode secret.
- Không log token hoặc password.
- Validate input.
- Không sử dụng file path do user nhập mà không kiểm tra.
- Tách config khỏi source code.

### Maintainability

- Không có main.py quá lớn.
- Module có trách nhiệm rõ.
- Có type hint.
- Có logging.
- Có config.
- Có test plan.

### Performance

- Camera đủ mượt.
- Drawing không có độ trễ rõ rệt.
- Không predict mỗi frame.
- Chỉ predict khi DONE.
- Không xử lý ảnh kích thước quá lớn không cần thiết.

## 8. MVP Technical Boundary

### Local Prototype

Được phép:

- Python.
- OpenCV.
- MediaPipe.
- NumPy.
- PyTorch hoặc TensorFlow.
- SQLite hoặc JSON.
- UI bằng cửa sổ OpenCV hoặc giao diện đơn giản.

Chưa cần:

- FastAPI.
- React.
- PostgreSQL.
- JWT.
- Docker production.
- Cloud deployment.

### Web MVP

Chỉ bắt đầu sau khi local prototype hoàn thành core flow.

## 9. Exit Criteria

Local MVP được coi là hoàn thành khi:

- Camera chạy ổn.
- Track được ngón trỏ.
- Vẽ được ký tự.
- Lưu được ảnh.
- Preprocessing đúng.
- Model nhận dạng được.
- Có confidence.
- Ghép được từ.
- Sửa được kết quả.
- Tra được nghĩa.
- Lưu được từ.
- Xem được lịch sử.
- Không lưu raw video.
- Có README.
- Có test plan.

## 10. Scope Control Rules

Mỗi tính năng mới phải trả lời:

1. Có phục vụ core flow không?
2. Có cần cho MVP không?
3. Có làm tăng rủi ro kỹ thuật không?
4. Có trì hoãn validation quan trọng không?
5. Có thể để giai đoạn sau không?

Nếu câu trả lời cho câu 1 là “không”, tính năng đó không nên vào MVP.
