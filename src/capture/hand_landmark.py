import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector(
    maxHands=1,
    detectionCon=0.7,
    minTrackCon=0.7
)

# MediaPipe 手部连接关系（21点骨架）
connections = [
    (0,1),(1,2),(2,3),(3,4),       # 拇指
    (0,5),(5,6),(6,7),(7,8),       # 食指
    (0,9),(9,10),(10,11),(11,12),  # 中指
    (0,13),(13,14),(14,15),(15,16),# 无名指
    (0,17),(17,18),(18,19),(19,20) # 小指
]

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        # ===== 画21个点 =====
        for i, (x, y, z) in enumerate(lmList):
            cv2.circle(img, (x, y), 6, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(i), (x+5, y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # ===== 强制画骨架 =====
        for a, b in connections:
            x1, y1, _ = lmList[a]
            x2, y2, _ = lmList[b]

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Hand 21 + Skeleton", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()