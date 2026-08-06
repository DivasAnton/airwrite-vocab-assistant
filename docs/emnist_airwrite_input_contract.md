# EMNIST - AirWrite Model Input Contract

## Mục đích

Sprint 8E đưa ảnh từ hai nguồn khác nhau về cùng một contract trước model, nhưng không ép hai nguồn
chạy cùng một chuỗi thao tác:

```text
EMNIST Letters -> EMNISTSourceAdapter ------┐
                                             ├-> ModelInputPreprocessor
AirWrite Canvas -> AirWriteCanvasAdapter ---┘
```

## Contract chung

| Thuộc tính | Giá trị |
| --- | --- |
| Prepared shape | `(28, 28)` |
| Model sample shape | `(28, 28, 1)` |
| Processed dtype/range | `uint8`, `0-255` |
| Normalized dtype/range | `float32`, `0.0-1.0` |
| Background | tối, giá trị chuẩn `0` |
| Foreground | sáng |
| Orientation | đứng và đọc được bằng mắt người |
| Normalization divisor | `255.0` |

`ModelInputPreprocessor` chỉ validate, copy và normalize ảnh đã được adapter chuẩn bị. Nó không crop,
resize, transpose, threshold hoặc tự invert.

## EMNIST Letters

`EMNISTSourceAdapter` nhận `(28,28)` hoặc `(28,28,1)` `uint8`, squeeze channel nếu cần và áp dụng
đúng một orientation transform theo cấu hình:

```text
EMNIST_TRANSPOSE_IMAGES=true  -> transpose
EMNIST_TRANSPOSE_IMAGES=false -> none
```

Adapter không crop, resize, binary threshold, center, augment hoặc map label. Grayscale được bảo
toàn. Intensity không tự invert; polarity và orientation là hai vấn đề độc lập.

EMNIST Letters có 26 letter identities. Các dạng chữ hoa và chữ thường của cùng chữ được gộp vào
một identity, nên Sprint 8E không tạo 52 class và chưa thêm case mode cho runtime.

## AirWrite Canvas

`AirWriteCanvasAdapter` giữ quy trình Sprint 8:

```text
validate -> grayscale -> binary mask -> full-foreground bbox
-> crop grayscale -> aspect-ratio resize -> center -> uint8 28x28
```

Binary mask chỉ tìm foreground và bounding box. Ảnh grayscale là dữ liệu được crop/resize, giúp giữ
anti-aliasing thay vì biến output thành ảnh nhị phân. Bounding box bao toàn bộ foreground, không chỉ
connected component lớn nhất, nên dấu chấm `i/j` và các nét tách rời được giữ lại.

## Compatibility

`HandwritingPreprocessor.process(image)` vẫn là façade cho AirWrite và giữ các field cũ:

- `processed_image`
- `normalized_image`
- `bounding_box`
- `original_shape`
- `cropped_shape`
- `foreground_pixel_count`
- `debug_images`

Metadata mới gồm `source`, `orientation_transform` và `foreground_ratio`. Sprint 10-11 không cần
đổi cách gọi. Model v0.1.0 và inference tensor adapter không bị thay trong Sprint 8E.

## Ownership

- Chỉ `EMNISTSourceAdapter` được phép sửa orientation EMNIST.
- Chỉ `AirWriteCanvasAdapter` crop/resize canvas AirWrite.
- Chỉ `ModelInputPreprocessor` normalize output chung.
- `PreprocessingAuditor` chỉ đo và xuất report; không tham gia prediction.
