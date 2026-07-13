# Product Vision

## 1. Tên sản phẩm

**AirWrite Vocabulary Assistant**

Tên thay thế có thể sử dụng trong một số ngữ cảnh:

**AirWrite Vocabulary Translator**

Tên được ưu tiên là **AirWrite Vocabulary Assistant** vì sản phẩm không chỉ dịch từ mà còn hỗ trợ nhận dạng, sửa, lưu và ôn tập từ vựng.

## 2. Product Vision Statement

> AirWrite Vocabulary Assistant là trợ lý học từ vựng sử dụng Computer Vision và air-writing, giúp người học tiếng Anh tra cứu, sửa, lưu và ôn tập từ mới khi đang xem phim hoặc video mà không cần liên tục chuyển tab và nhập từ bằng bàn phím.

## 3. Định vị sản phẩm

Định vị chính:

> A computer vision vocabulary learning assistant for movie and video learners.

Sản phẩm không được định vị là:

- Một bản demo MediaPipe.
- Một ứng dụng vẽ trong không khí.
- Một bài tập nhận dạng chữ viết tay.
- Một công cụ dịch đơn thuần.
- Một sản phẩm chỉ để trình diễn Computer Vision.

Sản phẩm phải được định vị là:

> AI-powered vocabulary assistant using air-writing interaction.

## 4. Người dùng mục tiêu

Nhóm người dùng đầu tiên:

- Người Việt học tiếng Anh.
- Thường xuyên học qua YouTube, phim hoặc video.
- Sử dụng laptop hoặc máy tính có webcam.
- Trình độ tiếng Anh từ cơ bản đến trung cấp.
- Muốn tra từ nhanh mà không làm gián đoạn quá trình học.
- Muốn lưu lại từ đã tra để ôn tập sau.

## 5. Vấn đề cốt lõi

Khi gặp từ mới trong video, người học thường phải:

1. Tạm dừng video.
2. Nhớ hoặc xem lại cách viết của từ.
3. Chuyển sang tab khác.
4. Mở từ điển hoặc công cụ dịch.
5. Nhập từ bằng bàn phím.
6. Đọc nghĩa.
7. Ghi chú nếu muốn lưu lại.
8. Quay lại video.

Quy trình này:

- Tốn thời gian.
- Làm mất tập trung.
- Phá vỡ mạch nội dung.
- Khiến người học bỏ qua từ mới.
- Không tạo được thói quen lưu và ôn tập từ.

## 6. Giải pháp

AirWrite cho phép người dùng:

1. Mở camera.
2. Giơ tay trước camera.
3. Viết từng ký tự bằng đầu ngón trỏ.
4. Để hệ thống dựng lại nét viết trên canvas.
5. Nhận dạng ký tự.
6. Ghép ký tự thành từ.
7. Sửa kết quả nếu model nhận sai.
8. Tra nghĩa và ví dụ.
9. Lưu từ vào kho từ vựng.
10. Ôn tập lại sau.

Core flow:

```text
Write → Recognize → Correct → Translate → Save → Review
```

## 7. Giá trị cốt lõi

Giá trị chính của sản phẩm không nằm ở việc “nhận dạng được chữ trong không khí”.

Giá trị chính là:

- Giảm số thao tác khi tra từ.
- Giữ người dùng trong ngữ cảnh học qua video.
- Biến một lần tra từ thành dữ liệu học tập có thể ôn lại.
- Kết nối interaction bằng camera với vocabulary workflow.
- Tạo trải nghiệm tra từ nhanh, trực quan và ít gián đoạn hơn.

## 8. Tầm nhìn dài hạn

Lộ trình sản phẩm:

```text
Local Prototype
→ Web MVP
→ PWA
→ Chrome Extension
→ Mobile App
→ Business-ready Product
```

Trong dài hạn, AirWrite có thể phát triển thành:

- Công cụ học từ vựng khi xem video.
- Floating widget trên YouTube hoặc Netflix.
- Kho từ vựng đồng bộ nhiều thiết bị.
- Hệ thống flashcard và spaced repetition.
- Công cụ xuất từ sang Anki hoặc CSV.
- Nền tảng B2B cho trung tâm tiếng Anh.
- Hệ thống theo dõi tiến độ học từ vựng.

## 9. Nguyên tắc sản phẩm

Thứ tự ưu tiên:

```text
Correctness
> User Experience
> Privacy/Security
> Maintainability
> Performance
> Business Value
> Convenience
```

Các nguyên tắc bắt buộc:

- Không gọi một bản demo là sản phẩm thương mại.
- Không coi model là luôn đúng.
- Recognition phải đi cùng correction flow.
- Không lưu raw video nếu không cần.
- Không gửi toàn bộ camera stream lên server.
- Không thêm business features khi core flow chưa ổn.
- Không ưu tiên công nghệ phức tạp hơn giá trị người dùng.
- Không mở rộng scope trước khi kiểm chứng interaction cốt lõi.

## 10. Mục tiêu MVP

MVP phải chứng minh được rằng:

1. Người dùng có thể viết ký tự trong không khí.
2. Hệ thống có thể dựng lại nét viết ổn định.
3. Model có thể nhận dạng ký tự từ canvas.
4. Người dùng có thể sửa kết quả.
5. Hệ thống có thể ghép thành từ.
6. Từ có thể được tra nghĩa và lưu lại.
7. Quy trình đủ thuận tiện để người dùng muốn thử lại.

## 11. Những gì chưa làm trong MVP

- Nhận dạng cả từ trong một lần viết.
- Mobile App.
- Chrome Extension.
- Payment.
- Subscription.
- Multi-language.
- Dashboard giáo viên.
- Social features.
- AI tutor.
- Fine-tune mô hình Transformer lớn.
- Đồng bộ nhiều thiết bị.

## 12. Tiêu chí tầm nhìn được xác nhận

Product Vision được coi là đủ rõ khi một người mới có thể trả lời:

- Sản phẩm dành cho ai?
- Vấn đề là gì?
- Tại sao air-writing được sử dụng?
- Sản phẩm khác air-writing demo ở đâu?
- Core flow là gì?
- MVP có gì?
- MVP chưa có gì?
- Tại sao correction flow là bắt buộc?
- Privacy được xử lý theo nguyên tắc nào?
