import pickle
import cv2
import mediapipe as mp
import numpy as np

# ── Load model ───────────────────────────────────────────────────────────────
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# ── Labels: 36 classes — A-Z (0-25) then 0-9 (26-35) ────────────────────────
labels_dict = {i: chr(ord('A') + i) for i in range(26)}
labels_dict.update({26 + i: str(i) for i in range(10)})

# ── MediaPipe 0.10+ new API (no mp.solutions) ────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH = 'hand_landmarker.task'

# Hand connections for drawing (21 landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit()

print("Sign Language Recognition running. Press 'Q' to quit.")

# ── Fullscreen window setup (must be before the loop) ────────────────────────
cv2.namedWindow('Sign Language Recognition', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sign Language Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = landmarker.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Draw connections
            for conn in HAND_CONNECTIONS:
                x0 = int(landmarks[conn[0]].x * W)
                y0 = int(landmarks[conn[0]].y * H)
                x1 = int(landmarks[conn[1]].x * W)
                y1 = int(landmarks[conn[1]].y * H)
                cv2.line(frame, (x0, y0), (x1, y1), (200, 200, 200), 2)

            # Draw landmark dots
            for lm in landmarks:
                cx, cy = int(lm.x * W), int(lm.y * H)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Build feature vector
            x_ = [lm.x for lm in landmarks]
            y_ = [lm.y for lm in landmarks]

            data_aux = []
            for lm in landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                proba      = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(proba) * 100

                predicted_char = labels_dict.get(int(prediction[0]), '?')

                # Bounding box
                x1b = max(0, int(min(x_) * W) - 20)
                y1b = max(0, int(min(y_) * H) - 20)
                x2b = min(W, int(max(x_) * W) + 20)
                y2b = min(H, int(max(y_) * H) + 20)

                cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 200, 0), 3)
                cv2.putText(frame,
                            f'{predicted_char}  ({confidence:.0f}%)',
                            (x1b, max(y1b - 15, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 0), 3, cv2.LINE_AA)

        cv2.putText(frame, "Press Q to quit", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()