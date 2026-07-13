# Architecture Overview

## 1. Mục tiêu kiến trúc

Kiến trúc của AirWrite phải:

- Tách trách nhiệm rõ ràng.
- Dễ test.
- Dễ debug.
- Có thể phát triển từ local prototype lên web MVP.
- Không trộn camera, gesture, model và database.
- Ưu tiên privacy.
- Hỗ trợ correction flow.
- Có thể thay model mà không sửa toàn bộ hệ thống.

## 2. High-Level Flow

```text
Camera Stream
    ↓
Hand Detector
    ↓
Finger Tracker
    ↓
Gesture Detector
    ↓
State Machine
    ↓
Stroke Manager
    ↓
Virtual Canvas
    ↓
Image Preprocessor
    ↓
Character Predictor
    ↓
Word Builder
    ↓
Correction Flow
    ↓
Dictionary Service
    ↓
Vocabulary Store
```

## 3. Nguyên tắc quan trọng

Camera frame không đi trực tiếp vào character model.

Luồng đúng:

```text
Camera
→ Hand Landmarks
→ Finger Coordinates
→ Stroke
→ Canvas Image
→ Preprocessing
→ Model
```

Lý do:

- Chữ không tồn tại thật trong camera frame.
- Camera chỉ ghi lại chuyển động tay.
- Hệ thống phải dựng lại nét viết.
- Model nhận dạng ảnh nét vẽ, không nhận dạng bàn tay.

## 4. Layer Responsibilities

## 4.1. Camera Layer

Trách nhiệm:

- Mở camera.
- Đọc frame.
- Flip frame.
- Kiểm soát FPS.
- Release camera.
- Xử lý lỗi camera.

Không chịu trách nhiệm:

- Detect gesture.
- Vẽ canvas.
- Predict ký tự.
- Lưu vocabulary.

## 4.2. Vision Layer

Trách nhiệm:

- Detect bàn tay.
- Trích xuất landmarks.
- Lấy landmark ngón trỏ.
- Convert tọa độ.
- Smoothing.
- Detect gesture.

Không chịu trách nhiệm:

- Dictionary lookup.
- Database.
- Model training.
- Vocabulary business logic.

## 4.3. Drawing Layer

Trách nhiệm:

- Quản lý stroke.
- Previous point.
- Current point.
- Vẽ line.
- Clear canvas.
- Kiểm tra empty canvas.
- Save canvas.

Không chịu trách nhiệm:

- Camera permission.
- Gesture classification.
- Model inference.
- Translation.

## 4.4. Preprocessing Layer

Trách nhiệm:

- Crop.
- Bounding box.
- Padding.
- Resize.
- Normalize.
- Convert channel.
- Validate input.

Không chịu trách nhiệm:

- Load database.
- Save vocabulary.
- Camera.
- Gesture.

## 4.5. Recognition Layer

Trách nhiệm:

- Load model.
- Validate model input.
- Predict label.
- Predict confidence.
- Map index sang ký tự.
- Quản lý model version.

Không chịu trách nhiệm:

- Word meaning.
- User correction.
- Database transaction.

## 4.6. Word Builder Layer

Trách nhiệm:

- Append character.
- Delete last character.
- Clear word.
- Confirm word.
- Normalize word.

Không chịu trách nhiệm:

- Model inference.
- Translation.
- Storage.

## 4.7. Correction Layer

Trách nhiệm:

- Hiển thị predicted result.
- Nhận corrected result.
- Gợi ý từ gần đúng.
- Confirm final word.
- Lưu predicted và corrected text nếu logging được phép.

## 4.8. Vocabulary Layer

Trách nhiệm:

- Dictionary lookup.
- Meaning.
- Part of speech.
- Phonetic.
- Example.
- Translation.
- Save vocabulary.
- Search vocabulary.
- Review vocabulary.

## 4.9. Storage Layer

Trách nhiệm:

- CRUD.
- Query.
- Duplicate handling.
- Transaction.
- Persistence.

Không chứa business logic.

## 4.10. UI Layer

Trách nhiệm:

- Camera preview.
- Canvas preview.
- Current state.
- Predicted character.
- Word buffer.
- Correction UI.
- Meaning.
- Save action.
- Error message.

## 5. Local Prototype Architecture

```text
app/
├── main.py
├── camera/
│   ├── camera_stream.py
│   └── frame_processor.py
├── vision/
│   ├── hand_detector.py
│   ├── finger_tracker.py
│   └── gesture_detector.py
├── drawing/
│   ├── canvas.py
│   ├── stroke_manager.py
│   └── image_saver.py
├── preprocessing/
│   └── handwriting_preprocessor.py
├── recognition/
│   ├── character_model.py
│   ├── predictor.py
│   └── label_mapping.py
├── vocabulary/
│   ├── dictionary_service.py
│   └── vocabulary_store.py
└── utils/
    ├── config.py
    └── logger.py
```

## 6. Runtime Flow

```text
main.py
→ CameraStream.read()
→ HandDetector.detect(frame)
→ FingerTracker.get_index_tip(landmarks)
→ GestureDetector.detect(landmarks)
→ StateMachine.update(gesture)
→ StrokeManager.update(point, state)
→ Canvas.draw()
→ khi DONE:
    → Canvas.export()
    → HandwritingPreprocessor.process()
    → Predictor.predict()
    → WordBuilder.append()
```

## 7. State Machine

```text
IDLE
→ READY
→ WRITING
→ PAUSED
→ DONE
→ READY
```

Các nhánh:

```text
WRITING → CLEAR → READY
ANY STATE → ERROR
ERROR → IDLE hoặc RETRY
```

Quy tắc:

- Chỉ WRITING mới vẽ.
- Khi mất tay, previous point phải reset.
- DONE chỉ predict một lần.
- CLEAR phải xóa canvas và reset stroke.
- ERROR không được tiếp tục gọi model.

## 8. Data Flow

### Camera Data

- Tồn tại trong memory.
- Không lưu raw frame.
- Không gửi server.

### Landmark Data

- Có thể xử lý trong memory.
- Không cần lưu mặc định.

### Canvas Image

- Có thể lưu cho debug hoặc dataset nếu có chủ đích.
- Không tự động lưu production nếu không cần.

### Prediction Data

Có thể gồm:

- Predicted character.
- Confidence.
- Model version.
- Timestamp.

### Vocabulary Data

Có thể gồm:

- Word.
- Meaning.
- Part of speech.
- Phonetic.
- Example.
- Created at.
- Review metadata.

## 9. Web MVP Architecture

### Frontend

- React hoặc Next.js.
- TypeScript.
- Browser camera.
- HTML Canvas.
- Correction UI.
- Vocabulary dashboard.

### Backend

- FastAPI.
- Pydantic.
- Service layer.
- Repository layer.
- ML inference service.
- Vocabulary service.
- Auth service.

### Database

- PostgreSQL hoặc MySQL.
- Users.
- Vocabularies.
- Saved words.
- Recognition logs.
- Correction logs.
- Review history.

### Flow

```text
Browser Camera
→ Client-side Hand Tracking
→ HTML Canvas
→ Canvas Image
→ FastAPI Prediction API
→ Prediction Response
→ Correction UI
→ Vocabulary API
→ Database
```

## 10. Privacy Architecture

Ưu tiên:

```text
Camera processing on client/local
→ retain only landmarks or canvas
→ send only canvas when needed
```

Không làm:

```text
Camera stream
→ continuous upload to backend
```

Privacy controls:

- Camera indicator.
- Permission request.
- Stop camera.
- Clear local data.
- Delete saved data.
- Retention policy.

## 11. Error Boundaries

Mỗi module phải trả lỗi rõ ràng:

- CameraOpenError.
- HandNotDetected.
- EmptyCanvasError.
- InvalidImageShape.
- ModelLoadError.
- PredictionError.
- DictionaryNotFound.
- StorageError.

Không nên để exception không kiểm soát đi thẳng tới UI.

## 12. Logging

Logging nên có:

- Application start.
- Camera opened.
- Model loaded.
- Prediction latency.
- Error type.
- Save result.

Không log:

- Raw video.
- Password.
- Token.
- Secret.
- Dữ liệu nhạy cảm.
- Ảnh canvas production nếu chưa có consent.

## 13. Testing Strategy

### Unit Tests

- Gesture detector.
- Canvas.
- Preprocessing.
- Word builder.
- Dictionary service.

### Integration Tests

- Canvas → preprocessing → predictor.
- Word builder → correction → dictionary.
- Vocabulary service → storage.

### End-to-End Tests

```text
Write
→ Predict
→ Correct
→ Translate
→ Save
```

## 14. Architecture Exit Criteria

Kiến trúc Sprint 0 được coi là đủ khi:

- Mỗi layer có trách nhiệm rõ.
- Camera frame không đi thẳng vào model.
- Có state machine.
- Có correction layer.
- Có privacy boundary.
- Có local flow.
- Có web migration path.
- Có test boundary.
- Không có main.py chứa toàn bộ logic.
