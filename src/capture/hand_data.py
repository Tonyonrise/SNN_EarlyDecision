import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import os

# ===== 初始化 =====
cap = cv2.VideoCapture(0)

detector = HandDetector(
    maxHands=1,
    detectionCon=0.7,
    minTrackCon=0.7
)

# 数据存储
recording = False
sequence = []
label = 4  # 你可以手动改类别
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

print("Press 's' start, 'e' end & save, 'q' quit")

# ===== 主循环 =====
while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # 21 points

        # 提取 21×3
        frame_data = np.array(lmList)

        if recording:
            sequence.append(frame_data)

        # 可视化21点 + 骨架
        for i, (x, y, z) in enumerate(lmList):
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, str(i), (x+3, y+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 状态显示
    if recording:
        cv2.putText(img, "RECORDING...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Data Collection", img)

    key = cv2.waitKey(1) & 0xFF

    # ===== start recording =====
    if key == ord('s'):
        recording = True
        sequence = []
        print("Start recording...")

    # ===== stop + save =====
    if key == ord('e'):
        if recording and len(sequence) > 5:
            recording = False

            sequence_np = np.array(sequence)  # (T,21,3)

            filename = f"{save_dir}/sample_{int(time.time())}_label_{label}.npy"
            np.save(filename, sequence_np)

            print(f"Saved: {filename}, shape={sequence_np.shape}")

        else:
            recording = False
            print("Recording too short or empty")

    # ===== quit =====
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()