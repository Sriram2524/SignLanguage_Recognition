import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe 0.10+ new API ──────────────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

MODEL_PATH = 'hand_landmarker.task'   # already in your project folder

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,                        # single hand → always 42 features
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

DATA_DIR = './data'
data   = []
labels = []
skipped = 0

with HandLandmarker.create_from_options(options) as landmarker:
    for dir_ in sorted(os.listdir(DATA_DIR), key=lambda x: int(x)):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_file in os.listdir(dir_path):
            img_bgr = cv2.imread(os.path.join(dir_path, img_file))
            if img_bgr is None:
                skipped += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = landmarker.detect(mp_image)

            if not result.hand_landmarks or len(result.hand_landmarks) != 1:
                skipped += 1
                continue

            landmarks = result.hand_landmarks[0]   # 21 landmarks
            x_ = [lm.x for lm in landmarks]
            y_ = [lm.y for lm in landmarks]

            data_aux = []
            for lm in landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)
            else:
                skipped += 1

print(f'Dataset: {len(data)} samples, {skipped} skipped.')

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print('Saved to data.pickle')