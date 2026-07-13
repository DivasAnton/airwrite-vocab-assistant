# AirWrite Vocabulary Assistant

> AI-powered vocabulary assistant using air-writing interaction.

## 1. Tổng quan

AirWrite Vocabulary Assistant là một sản phẩm AI hỗ trợ người học tiếng Anh tra cứu, sửa, lưu và ôn tập từ mới trong khi đang xem phim, video hoặc tài liệu tiếng Anh.

Thay vì phải tạm dừng nội dung, chuyển tab, mở từ điển và nhập từ bằng bàn phím, người dùng có thể:

1. Mở camera.
2. Giơ bàn tay trước camera.
3. Dùng ngón trỏ viết từng ký tự trong không khí.
4. Để hệ thống theo dõi chuyển động và dựng lại nét viết trên canvas.
5. Nhận dạng ký tự.
6. Ghép các ký tự thành từ.
7. Sửa kết quả nếu cần.
8. Tra nghĩa, từ loại, phiên âm và ví dụ.
9. Lưu từ vào kho từ vựng cá nhân.
10. Ôn tập lại sau.

## 2. Product Vision

AirWrite Vocabulary Assistant hướng tới trở thành một trợ lý học từ vựng dành cho người học ngoại ngữ qua video, sử dụng Computer Vision và air-writing để giảm thao tác tra từ và duy trì sự tập trung.

Định vị sản phẩm:

> A computer vision vocabulary learning assistant for movie and video learners.

Sản phẩm không được định vị chỉ là một bản demo viết trong không khí.

## 3. Vấn đề cần giải quyết

Người học tiếng Anh qua video thường phải:

- Tạm dừng video.
- Nhớ hoặc nhìn lại cách viết của từ.
- Chuyển sang tab khác.
- Mở từ điển hoặc công cụ dịch.
- Nhập từ bằng bàn phím.
- Đọc nghĩa.
- Ghi chú thủ công nếu muốn lưu.
- Quay lại video.

Quy trình này gây mất tập trung, tốn thao tác và khiến nhiều từ mới bị bỏ qua hoặc không được lưu để ôn tập.

## 4. Giải pháp đề xuất

AirWrite cung cấp một luồng liền mạch:

```text
Write → Recognize → Correct → Translate → Save → Review
```

Luồng xử lý kỹ thuật:

```text
Camera
→ Hand Detection
→ Finger Tracking
→ Gesture Detection
→ Air Drawing Canvas
→ Image Preprocessing
→ Character Recognition
→ Word Builder
→ Correction Flow
→ Translation
→ Vocabulary Storage
```

## 5. Người dùng mục tiêu ban đầu

Nhóm người dùng đầu tiên:

- Người Việt học tiếng Anh.
- Độ tuổi tham khảo: 18–26.
- Trình độ tiếng Anh từ A2 đến B1.
- Thường học qua YouTube, phim hoặc video.
- Sử dụng laptop hoặc máy tính có webcam.
- Có nhu cầu tra và lưu từ mới.
- Không muốn liên tục chuyển tab khi học.

## 6. MVP Scope

MVP đầu tiên cần đạt:

- Mở camera.
- Detect bàn tay.
- Track đầu ngón trỏ.
- Vẽ nét lên canvas.
- Lưu ảnh ký tự.
- Tiền xử lý ảnh.
- Nhận dạng từng ký tự.
- Trả confidence score.
- Ghép ký tự thành từ.
- Cho phép người dùng sửa kết quả.
- Tra nghĩa tiếng Việt.
- Hiển thị ví dụ tiếng Anh và tiếng Việt.
- Lưu từ vào lịch sử cục bộ.
- Xem lại danh sách từ đã lưu.

Chưa thuộc MVP:

- Nhận dạng nguyên từ trong một lần viết.
- Chrome Extension.
- Mobile App.
- Thanh toán và subscription.
- Đồng bộ nhiều thiết bị.
- Dashboard giáo viên.
- Hỗ trợ nhiều ngôn ngữ.
- Social features.

## 7. Nguyên tắc kiến trúc

- Không đưa toàn bộ logic vào một file `main.py`.
- Camera, vision, drawing, preprocessing, recognition và vocabulary phải được tách module.
- Camera frame không được đưa trực tiếp vào model nhận dạng chữ.
- Model chỉ nhận ảnh canvas sau khi đã crop, padding, resize và normalize.
- Không predict liên tục ở từng frame.
- Chỉ predict khi người dùng kết thúc một ký tự.
- Không lưu raw video.
- Không gửi toàn bộ video stream lên backend nếu không cần.
- Recognition phải luôn đi cùng correction flow.

## 8. Privacy Principles

- Camera được ưu tiên xử lý local hoặc client-side.
- Không lưu raw video của người dùng.
- Chỉ lưu ảnh canvas nếu có lý do rõ ràng.
- Nếu gửi dữ liệu lên backend, chỉ gửi ảnh canvas hoặc dữ liệu cần thiết.
- Không log token, password, secret hoặc dữ liệu camera.
- Người dùng phải biết khi camera đang hoạt động.
- Người dùng phải có quyền xóa dữ liệu đã lưu.

## 9. Roadmap tổng quát

### Giai đoạn 1 — Local Prototype

- Camera.
- Hand detection.
- Finger tracking.
- Gesture control.
- Air drawing.
- Preprocessing.
- Character recognition.
- Word builder.
- Translation.
- Local storage.
- Correction flow.

### Giai đoạn 2 — Web MVP

- React hoặc Next.js.
- Browser camera.
- HTML Canvas.
- FastAPI backend.
- ML inference API.
- User authentication.
- Vocabulary history theo user.

### Giai đoạn 3 — Product Expansion

- PWA.
- Chrome Extension.
- Flashcard.
- Spaced repetition.
- Deployment.
- Monitoring.
- Security.
- Freemium và subscription.

## 10. Tiêu chuẩn hoàn thành

### Mức demo kỹ thuật

- Camera hoạt động.
- Detect được tay.
- Track được ngón trỏ.
- Vẽ được nét lên canvas.

### Mức local product prototype

- Nhận dạng ký tự.
- Ghép thành từ.
- Cho phép sửa.
- Tra nghĩa.
- Hiển thị ví dụ.
- Lưu từ.
- Xem lịch sử.

### Mức sản phẩm thực tế

- Có web app.
- Có tài khoản người dùng.
- Có database.
- Có logging và error handling.
- Có privacy notice.
- Có deploy public.
- Có phản hồi từ người dùng thật.

## 11. Cấu trúc tài liệu Sprint 0

```text
docs/
├── product_vision.md
├── problem_statement.md
├── target_users.md
├── user_flow.md
├── mvp_scope.md
├── success_metrics.md
├── assumptions_and_risks.md
└── architecture_overview.md
```

## 12. Trạng thái hiện tại

```text
Current Sprint: Sprint 0 — Product Vision & Planning
Status: Draft completed
Next Sprint: Sprint 1 — Environment Setup / Git / Project Structure
```
