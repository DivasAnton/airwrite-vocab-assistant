# Problem Statement

## 1. Bối cảnh

Nhiều người học tiếng Anh thông qua:

- Phim.
- YouTube.
- Video học thuật.
- Video giải trí.
- Podcast có phụ đề.
- Khóa học trực tuyến.
- Tài liệu hiển thị trên màn hình.

Trong quá trình đó, người học thường xuyên gặp từ mới nhưng không muốn làm gián đoạn nội dung đang xem.

## 2. Quy trình hiện tại

Khi gặp từ mới, người dùng thường phải:

```text
Xem video
→ gặp từ mới
→ pause video
→ nhìn lại hoặc nhớ cách viết
→ chuyển tab
→ mở từ điển
→ nhập từ
→ đọc nghĩa
→ ghi chú thủ công nếu cần
→ chuyển lại video
→ tiếp tục xem
```

## 3. Pain Points

### 3.1. Quá nhiều thao tác

Người dùng phải thực hiện nhiều bước chỉ để tra một từ.

### 3.2. Mất tập trung

Việc rời khỏi video làm gián đoạn luồng suy nghĩ và giảm mức độ tập trung.

### 3.3. Từ mới bị bỏ qua

Khi phải thực hiện quá nhiều thao tác, người dùng thường chọn bỏ qua từ mới.

### 3.4. Không có quy trình lưu từ

Từ đã tra thường không được lưu lại hoặc được ghi chú rời rạc.

### 3.5. Không có quy trình ôn tập

Ngay cả khi đã ghi chú, người dùng cũng ít có hệ thống ôn tập lại.

### 3.6. Công cụ hiện tại tách rời nhau

Từ điển, ứng dụng ghi chú, flashcard và video thường nằm ở nhiều công cụ khác nhau.

## 4. Đối tượng gặp vấn đề

Nhóm người dùng chính:

- Người Việt học tiếng Anh.
- Học qua video hoặc phim trên laptop.
- Trình độ từ A2 đến B1.
- Thường gặp từ mới.
- Muốn tra từ nhanh.
- Chưa có hệ thống lưu và ôn từ tốt.

## 5. Hậu quả

Nếu vấn đề không được giải quyết:

- Người học bỏ qua nhiều từ mới.
- Việc học qua video kém hiệu quả.
- Người dùng mất mạch nội dung.
- Từ đã tra nhanh chóng bị quên.
- Người dùng không xây dựng được vocabulary history.
- Quá trình học trở nên rời rạc.

## 6. Problem Statement chính thức

> Người học tiếng Anh qua phim và video thường phải tạm dừng nội dung, chuyển sang từ điển, nhập từ mới, đọc nghĩa rồi quay lại video. Quy trình này gây mất tập trung, tốn thao tác và khiến nhiều từ mới không được tra hoặc lưu lại. AirWrite Vocabulary Assistant hướng tới giảm các bước đó bằng cách cho phép người dùng viết từ trước camera, nhận dạng, sửa, dịch và lưu từ trong một luồng liền mạch.

## 7. Job To Be Done

> Khi tôi gặp một từ mới trong lúc xem video, tôi muốn tra và lưu từ đó nhanh chóng mà không phải rời khỏi ngữ cảnh đang học, để tôi có thể tiếp tục xem và ôn lại từ sau này.

## 8. Giải pháp đề xuất

```text
Xem video
→ gặp từ mới
→ kích hoạt camera
→ viết từng ký tự
→ hệ thống nhận dạng
→ sửa hoặc xác nhận
→ tra nghĩa
→ lưu từ
→ quay lại nội dung
```

## 9. Giá trị cần kiểm chứng

Giải pháp chỉ có giá trị khi:

- Nhanh hơn hoặc ít gây gián đoạn hơn quy trình hiện tại.
- Người dùng có thể hoàn thành flow mà không bối rối.
- Recognition đủ tốt hoặc correction đủ nhanh.
- Người dùng thực sự lưu từ.
- Người dùng quay lại ôn tập.
- Camera không tạo cảm giác xâm phạm privacy.

## 10. Các giải pháp thay thế cần so sánh

AirWrite không phải giải pháp duy nhất. Cần so sánh với:

### Gõ bàn phím

Ưu điểm:

- Chính xác.
- Quen thuộc.
- Nhanh với người gõ tốt.

Nhược điểm:

- Phải chuyển focus.
- Có thể cần chuyển tab.
- Không tạo cảm giác liền mạch.

### Voice Input

Ưu điểm:

- Nhanh.
- Không cần gõ.

Nhược điểm:

- Khó dùng trong môi trường ồn.
- Người dùng có thể không biết phát âm.
- Có vấn đề privacy âm thanh.
- Không phù hợp khi đang xem phim cùng người khác.

### OCR từ phụ đề

Ưu điểm:

- Có thể tự động.

Nhược điểm:

- Không phải video nào cũng có phụ đề.
- Có thể vi phạm giới hạn kỹ thuật hoặc nền tảng.
- Không phù hợp với từ không hiển thị rõ.
- Không giải quyết được nhu cầu interaction chủ động.

### Air-writing

Ưu điểm:

- Trực quan.
- Không cần bàn phím.
- Có thể gắn với floating widget.
- Tạo interaction khác biệt.

Nhược điểm:

- Có thể chậm.
- Có thể mỏi tay.
- Recognition có thể sai.
- Cần camera.
- Có privacy concern.

## 11. Kết luận

Rủi ro lớn nhất không phải là “model có train được hay không”.

Rủi ro lớn nhất là:

> Người dùng có thật sự thấy air-writing tiện và đáng sử dụng hơn các phương thức hiện có hay không?

Do đó, toàn bộ local prototype phải được xây để kiểm chứng giả thuyết này, không chỉ để chứng minh công nghệ có thể chạy.
