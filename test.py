import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

while True:
    ret, frame = cap.read()  # Grab a frame
    if not ret:
        break
    
    cv2.imshow('Webcam Test', frame)  # Show it
    
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()