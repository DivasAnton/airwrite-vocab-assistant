# User Flow

## 1. Mục tiêu

Tài liệu này mô tả:

- Luồng hiện tại của người dùng.
- Luồng đề xuất với AirWrite.
- Trạng thái chính của sản phẩm.
- Các nhánh lỗi và correction.
- Luồng dữ liệu mức cao.

## 2. Current User Journey

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

### Pain Points

- Nhiều thao tác.
- Mất focus.
- Dễ quên từ.
- Dễ bỏ qua từ.
- Không có lịch sử thống nhất.
- Không có quy trình ôn tập.

## 3. Proposed User Journey

```text
Xem video
→ gặp từ mới
→ mở AirWrite
→ cấp quyền camera
→ đưa tay vào vùng nhận diện
→ kích hoạt chế độ viết
→ viết từng ký tự
→ hệ thống nhận dạng từng ký tự
→ ghép ký tự thành từ
→ người dùng xác nhận hoặc sửa
→ hệ thống tra nghĩa
→ hiển thị nghĩa và ví dụ
→ người dùng lưu từ
→ tiếp tục xem video
```

## 4. Core Product Flow

```text
Write
→ Recognize
→ Correct
→ Translate
→ Save
→ Review
```

### Write

Người dùng viết ký tự bằng đầu ngón trỏ.

### Recognize

Hệ thống dựng ảnh canvas, preprocess và predict.

### Correct

Người dùng sửa nếu kết quả sai.

### Translate

Hệ thống trả nghĩa, từ loại, phiên âm và ví dụ.

### Save

Người dùng lưu từ vào vocabulary history.

### Review

Người dùng xem lại hoặc ôn tập từ đã lưu.

## 5. Detailed Local Prototype Flow

```text
Khởi động ứng dụng
→ kiểm tra camera
→ camera hoạt động?
    ├── Không → hiển thị lỗi và hướng dẫn
    └── Có
        → đọc frame
        → detect hand
        → tìm landmark ngón trỏ
        → xác định gesture
        → cập nhật state
        → nếu WRITING:
            → lấy current point
            → nối previous point với current point
            → vẽ lên canvas
        → nếu DONE:
            → kiểm tra canvas rỗng
            → crop
            → padding
            → resize
            → normalize
            → predict character
            → trả label + confidence
            → thêm character vào word buffer
        → nếu CLEAR:
            → xóa canvas hiện tại
        → nếu CONFIRM WORD:
            → correction flow
            → dictionary lookup
            → hiển thị kết quả
            → lưu nếu user yêu cầu
```

## 6. State Machine

Các trạng thái đề xuất:

### IDLE

- Camera hoạt động.
- Chưa sẵn sàng viết.
- Không vẽ.

### READY

- Bàn tay đã được nhận diện.
- Hệ thống chờ gesture bắt đầu.

### WRITING

- Đầu ngón trỏ đang tạo nét.
- Canvas được cập nhật.

### PAUSED

- Tạm dừng nét hiện tại.
- Không vẽ.
- Có thể tiếp tục.

### DONE

- Người dùng kết thúc ký tự.
- Hệ thống chạy preprocessing và prediction.

### CLEAR

- Xóa canvas ký tự hiện tại.
- Reset điểm trước đó.

### ERROR

- Camera lỗi.
- Không detect được tay trong thời gian dài.
- Input không hợp lệ.
- Model hoặc file cấu hình lỗi.

## 7. Character Flow

```text
User viết ký tự
→ hoàn thành ký tự
→ save canvas snapshot
→ preprocess
→ predict
→ label + confidence
→ hiển thị dự đoán
→ user xác nhận hoặc sửa
→ append vào word buffer
→ clear canvas ký tự
→ viết ký tự tiếp theo
```

Ví dụ:

```text
c → o → o → k
```

Word buffer:

```text
["c", "o", "o", "k"] → "cook"
```

## 8. Correction Flow

```text
Predicted word: c00k
→ hiển thị confidence
→ tạo suggestions
→ cook
→ look
→ book
→ user chọn cook
→ corrected word = cook
→ dictionary lookup
→ save corrected word
```

Các cách correction:

- Chọn từ gợi ý.
- Sửa trực tiếp bằng bàn phím.
- Xóa ký tự cuối.
- Viết lại một ký tự.
- Viết lại toàn bộ từ.

## 9. Translation Flow

```text
Corrected word
→ normalize input
→ validate word
→ dictionary lookup
→ found?
    ├── Có
    │   → meaning
    │   → part of speech
    │   → phonetic
    │   → example EN
    │   → example VI
    └── Không
        → hiển thị not found
        → cho phép chỉnh sửa
        → fallback service nếu có
```

## 10. Save Vocabulary Flow

```text
User chọn Save
→ validate vocabulary
→ kiểm tra duplicate
→ nếu chưa tồn tại:
    → tạo record
→ nếu đã tồn tại:
    → cập nhật metadata phù hợp
→ lưu source
→ lưu created_at
→ hiển thị success
```

Không lưu từ chưa được người dùng xác nhận.

## 11. Error Flows

### Camera không mở được

```text
Open camera
→ failed
→ hiển thị lỗi
→ hướng dẫn kiểm tra permission hoặc thiết bị
→ cho phép retry
```

### Không detect được tay

```text
No hand detected
→ giữ state an toàn
→ không vẽ
→ hiển thị hướng dẫn đưa tay vào vùng camera
```

### Canvas rỗng

```text
DONE
→ canvas empty
→ không predict
→ hiển thị yêu cầu viết lại
```

### Model confidence thấp

```text
Prediction confidence < threshold
→ hiển thị cảnh báo
→ đề xuất viết lại hoặc sửa thủ công
```

### Dictionary không tìm thấy từ

```text
Not found
→ không lưu nghĩa giả
→ cho phép chỉnh sửa từ
→ dùng fallback nếu được cấu hình
```

## 12. Web MVP Flow

```text
Browser
→ request camera permission
→ camera preview
→ browser hand tracking hoặc gửi canvas
→ HTML Canvas
→ character image
→ FastAPI inference endpoint
→ prediction response
→ correction UI
→ vocabulary API
→ user database
```

## 13. Chrome Extension Flow

```text
User đang xem YouTube hoặc Netflix
→ mở floating widget
→ camera mini panel
→ viết từ
→ nhận dạng
→ correction
→ translation
→ lưu cloud vocabulary
→ đóng widget
→ tiếp tục xem
```

## 14. UX Principles

- Trạng thái hiện tại phải rõ.
- Không có quá nhiều gesture.
- Luôn có cách sửa thủ công.
- Không predict khi người dùng chưa kết thúc.
- Không lưu tự động nếu chưa xác nhận.
- Không để camera hoạt động âm thầm.
- Không bắt người dùng viết lại toàn bộ nếu chỉ sai một ký tự.

## 15. Definition of Done

User flow được coi là đủ rõ khi:

- Có happy path.
- Có error path.
- Có correction path.
- Có state machine.
- Có local prototype flow.
- Có web flow.
- Có extension flow.
- Có quy tắc privacy.
- Có quy tắc save.
