import cv2
cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/live")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
