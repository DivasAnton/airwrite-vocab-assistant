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
 
## Sprint 2 Webcam Prototype

Sprint 2 adds the first OpenCV camera layer for AirWrite.

- `CameraStream` opens the configured webcam, requests width, height, and FPS, reads valid frames, reports actual camera settings, and releases the camera safely.
- `FrameProcessor` validates frames, mirrors the preview when enabled, calculates processing FPS, and draws prototype debug information.
- `python -m app.main` starts the webcam prototype from the project root.
- Quit controls: `Q`, `q`, `ESC`, or closing the OpenCV window.
- Raw camera frames are not saved, recorded, or uploaded.

Camera configuration is read from `.env` or `.env.example`:

```env
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30
CAMERA_MIRROR=true
CAMERA_WINDOW_NAME=AirWrite Camera
SHOW_FPS=true
```

Quality checks for this sprint:

```powershell
ruff format .
ruff check .
pytest
mypy app
```

Manual camera testing is documented in `docs/sprint_2_camera_test_plan.md`.

## Sprint 3 Hand Detection

Sprint 3 adds MediaPipe Hand Landmarker detection on top of the Sprint 2 webcam
prototype.

- `HandDetector` receives mirrored OpenCV BGR frames, converts them to RGB, runs
  MediaPipe in VIDEO mode, and returns project-owned result objects.
- `HandDetectionResult` keeps hand landmarks, world landmarks, handedness, and
  timestamps out of `main.py`.
- `HandLandmarkRenderer` draws landmarks, hand connections, and handedness for
  debug display.
- `main.py` now creates the detector once, keeps timestamps increasing, displays
  hand count/status, and closes both camera and landmarker during cleanup.
- Sprint 3 does not add finger tracking, gestures, canvas drawing, frame saving,
  or camera upload.

Hand detection configuration:

```env
HAND_LANDMARKER_MODEL_PATH=models/hand_landmarker.task
HAND_NUM_HANDS=1
HAND_MIN_DETECTION_CONFIDENCE=0.5
HAND_MIN_PRESENCE_CONFIDENCE=0.5
HAND_MIN_TRACKING_CONFIDENCE=0.5
DRAW_HAND_LANDMARKS=true
DRAW_HAND_CONNECTIONS=true
DRAW_HANDEDNESS=true
```

Model setup:

1. Download a compatible MediaPipe Hand Landmarker `.task` model.
2. Place it at `models/hand_landmarker.task`.
3. Keep the path relative to the project. Do not use an absolute local path.

Manual camera testing is documented in `docs/sprint_3_hand_detection_test_plan.md`.

## Sprint 4 Index Finger Tracking

Sprint 4 extracts landmark 8 from the detected hand and turns it into a stable
debug cursor.

- `IndexFingerTracker` selects the first detected hand, reads landmark 8, converts
  normalized coordinates to frame pixels, clamps the point to frame boundaries,
  applies exponential smoothing, and resets when the hand is lost.
- `FingerTrackingResult` keeps raw point, smoothed point, normalized point,
  timestamp, hand index, and landmark index in one immutable result object.
- `FingerTrackingRenderer` draws the raw and/or smoothed fingertip point for
  debugging.
- `main.py` only coordinates detector, hand renderer, finger tracker, finger
  renderer, and debug overlay.
- Sprint 4 does not change handedness labels, draw canvas lines, add gestures, or
  save raw camera frames.

Finger tracking configuration:

```env
INDEX_FINGER_LANDMARK_INDEX=8
FINGER_SMOOTHING_ALPHA=0.5
DRAW_RAW_FINGER_POINT=false
DRAW_SMOOTHED_FINGER_POINT=true
FINGER_POINT_RADIUS=8
```

## Sprint 5 Air Drawing Canvas

Sprint 5 turns the smoothed index finger point into a persistent virtual drawing
canvas.

- `AirCanvas` owns the independent NumPy canvas, draws anti-aliased line
  segments, clears content, and resets safely when frame size changes.
- `StrokeManager` owns previous/current point logic, skips the first point,
  resets when tracking is lost, and ignores unusually large jumps.
- `CanvasOverlayRenderer` overlays the canvas onto the camera preview without
  mutating the frame or canvas inputs.
- `main.py` now creates the canvas from the actual frame size, draws line
  segments from the smoothed finger point, shows an optional canvas window, and
  clears both canvas and stroke state with the configured key.
- Sprint 5 does not add gesture control, image saving, preprocessing, character
  recognition, prediction, or handedness changes.

Canvas configuration:

```env
CANVAS_BACKGROUND_COLOR=0,0,0
CANVAS_STROKE_COLOR=255,255,255
CANVAS_STROKE_THICKNESS=8
CANVAS_MAX_POINT_DISTANCE=120
CANVAS_OVERLAY_OPACITY=1.0
SHOW_CAMERA_WITH_CANVAS=true
SHOW_CANVAS_WINDOW=true
CANVAS_CLEAR_KEY=c
```

Manual canvas testing is documented in `docs/sprint_5_air_canvas_test_plan.md`.

## Sprint 6 Gesture Control And Drawing State

Sprint 6 separates hand shape detection from drawing state.

- `GestureDetector` classifies `INDEX_ONLY`, `OPEN_PALM`, `FIST`, `UNKNOWN`, and
  `NO_HAND` from hand landmarks without using handedness or thumb logic.
- `GestureStabilizer` accepts a gesture only after enough stable frames, reducing
  flicker from noisy landmarks.
- `DrawingStateMachine` owns transitions between `IDLE`, `READY`, `WRITING`,
  `PAUSED`, `DONE`, and `CLEAR`.
- `DrawingController` draws only in `WRITING`, resets strokes when leaving or
  entering writing, clears safely, and keeps canvas/stroke actions outside
  `main.py`.
- Keyboard fallback remains available: `Space` toggles writing/pause, `C` clears,
  and `D` marks done.
- Sprint 6 does not save images, run OCR, preprocess canvas data, translate text,
  or change handedness.

Gesture configuration:

```env
GESTURE_STABLE_FRAMES=5
GESTURE_COOLDOWN_MS=500
GESTURE_LOST_HAND_FRAMES=10
ENABLE_GESTURE_CONTROL=true
ENABLE_KEYBOARD_FALLBACK=true
DRAW_GESTURE_LABEL=true
DRAW_STATE_LABEL=true
CLEAR_HOLD_MS=1500
ENABLE_CLEAR_GESTURE=false
FINGER_EXTENSION_MARGIN=0.02
```

Manual gesture testing is documented in `docs/sprint_6_gesture_control_test_plan.md`.

Manual tracking testing is documented in `docs/sprint_4_finger_tracking_test_plan.md`.

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
