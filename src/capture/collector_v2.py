import cv2
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# ======================
# Camera + Detector
# ======================
cap = cv2.VideoCapture(0)

detector = HandDetector(
    maxHands=1,
    detectionCon=0.7,
    minTrackCon=0.7
)

# ======================
# label mapping
# ======================
label_map = {
    ord('1'): "left",
    ord('2'): "right",
    ord('3'): "up",
    ord('4'): "down",
    ord('5'): "wave"
}

SAVE_PATH = "dataset/raw"
os.makedirs(SAVE_PATH, exist_ok=True)

current_label = None
recording = False
sequence = []

# ======================
# get next index
# ======================
def get_next_index(folder, label):
    if not os.path.exists(folder):
        return 1

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    if len(files) == 0:
        return 1

    return len(files) + 1


print("Collector V2 (with indexing) Started")

# ======================
# loop
# ======================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        if recording:
            sequence.append(lmList)

    # ======================
    # UI
    # ======================
    cv2.putText(img, f"Label: {current_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(img, f"Recording: {recording}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(img, f"Frames: {len(sequence)}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Collector CVZone V2", img)

    key = cv2.waitKey(1) & 0xFF

    # ======================
    # select label
    # ======================
    if key in label_map:
        current_label = label_map[key]
        print("Label set:", current_label)

    # ======================
    # start recording
    # ======================
    elif key == ord('s'):
        if current_label is None:
            print("Select label first")
            continue

        recording = True
        sequence = []
        print("Recording started")

    # ======================
    # stop + save
    # ======================
    elif key == ord('e'):
        if recording:
            recording = False

            if len(sequence) < 10:
                print("Too short, discarded")
                continue

            folder = os.path.join(SAVE_PATH, current_label)
            os.makedirs(folder, exist_ok=True)

            idx = get_next_index(folder, current_label)

            filename = f"{current_label}_{idx:04d}.npy"
            path = os.path.join(folder, filename)

            np.save(path, np.array(sequence))

            print(f"Saved: {path} | frames={len(sequence)}")

            sequence = []

    # ======================
    # quit
    # ======================
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()