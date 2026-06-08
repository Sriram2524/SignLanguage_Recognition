import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 26 letters (A-Z) + 10 digits (0-9) = 36 classes
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(10)]
dataset_size = 100

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit()

for j, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'=== Collecting data for class [{class_name}] (folder {j}) ===')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame,
                    f'Class: {class_name}  |  Press "Q" to start!',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collect Images', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame,
                    f'Class: {class_name}  |  {counter}/{dataset_size}',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.imshow('Collect Images', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

    print(f'  Done collecting {dataset_size} images for [{class_name}]')

cap.release()
cv2.destroyAllWindows()
print('All classes collected successfully!')