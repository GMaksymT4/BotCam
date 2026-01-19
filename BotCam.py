import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'), 
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.5  # Lower for more detections, higher for fewer false positives
)

detector = vision.FaceDetector.create_from_options(options)


cap = cv2.VideoCapture(0)  # Open webcam
blur_enabled = False
box_enabled = False
outline_enabled = False
while True:
    ret, frame = cap.read()  # Grab frame
    if not ret:
        break
    
    # Step 1: Convert OpenCV frame (BGR) to MediaPipe Image (RGB implied)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Step 2: Get timestamp in milliseconds (VIDEO mode expects increasing timestamps)
    timestamp_ms = int(time.time() * 1000)
    
    # Step 3: Run detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    # Step 4: Draw boxes 
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x = int(bbox.origin_x-10)      # Starting x coordinate
        y = int(bbox.origin_y-100)      # Starting y coordinate
        w = int(bbox.width+10)         # Width of box
        h = int(bbox.height+100)        # Height of box     



        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            blur_enabled = not blur_enabled     # flip True ↔ False
            print("Blur:", "ON" if blur_enabled else "OFF")   
            
        if key == ord('e'):
            box_enabled = not box_enabled     # flip True ↔ False
            print("Box:", "ON" if box_enabled else "OFF")   
        if key == ord('p'):
            outline_enabled = not outline_enabled     # flip True ↔ False
            print("Outline:", "ON" if outline_enabled else "OFF")


        else:
            pass

        if blur_enabled:
            face_region = frame[y:y+h, x:x+w]
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(face_region, (35, 35), 25)
                #99,99,7
                #35,35,25
                # Odd kernel size (99 is very strong), 30 = sigma (strength)
            frame[y:y+h, x:x+w] = blurred_face
            cv2.putText(frame, f'Blur: {blur_enabled}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if box_enabled:
            face_region = frame[y:y+h, x:x+w]
            # Apply black box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Green box(0,225,0), thicker(3)
            cv2.putText(frame, f'Box: {box_enabled}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if outline_enabled:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box(0,225,0), thicker(3)
            cv2.putText(frame, f'Outline: {outline_enabled}', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
    
    # Optional: Show number of faces
    num_faces = len(detection_result.detections)
    cv2.putText(frame, f'Faces: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('MediaPipe Face Detection', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()