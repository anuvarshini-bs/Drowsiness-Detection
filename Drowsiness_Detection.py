import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

def eye_aspect_ratio(eye_indices, facial_landmarks, img_w, img_h):
    # Convert normalized landmarks to pixel coordinates
    pts = []
    for idx in eye_indices:
        landmark = facial_landmarks[idx]
        pts.append((int(landmark.x * img_w), int(landmark.y * img_h)))
    
    # Calculate distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    
    if C == 0:
        return 0, pts
    
    ear = (A + B) / (2.0 * C)
    return ear, pts

def mouth_aspect_ratio(mouth_indices, facial_landmarks, img_w, img_h):
    # Convert normalized landmarks to pixel coordinates
    pts = []
    for idx in mouth_indices:
        landmark = facial_landmarks[idx]
        pts.append(np.array([landmark.x * img_w, landmark.y * img_h]))
    
    # Calculate MAR
    A = np.linalg.norm(pts[3] - pts[9])  # 181 to 291
    B = np.linalg.norm(pts[2] - pts[10])  # 91 to 375
    C = np.linalg.norm(pts[0] - pts[6])  # 61 to 314
    
    if C == 0:
        return 0, pts
    
    mar = (A + B) / (2.0 * C)
    return mar, pts

def compute_face_distance(ref, current):
    dist = 0
    for i in range(min(len(ref), len(current))):
        dx = ref[i].x - current[i].x
        dy = ref[i].y - current[i].y
        dz = ref[i].z - current[i].z
        dist += np.sqrt(dx**2 + dy**2 + dz**2)
    return dist / min(len(ref), len(current))

thresh = 0.25
frame_check = 20
flag = 0

thresh_mar = 0.5
seizure_frame_check = 10
seizure_flag = 0

thresh_dist = 0.15  # Threshold for face distance
person_change_flag = 0
person_change_frame_check = 10

# MediaPipe Eye indices (Order: left/right corner, top-left, top-right, right/left corner, bottom-right, bottom-left)
# Right eye (from image perspective, user's right eye):
RIGHT_EYE = [33, 160, 158, 133, 153, 144] 
# Left eye (from image perspective, user's left eye):
LEFT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]

face_landmarks = None
reference_landmarks = None

def result_callback(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global face_landmarks
    face_landmarks = result.face_landmarks

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)
timestamp = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (600, 450))
        img_h, img_w, _ = frame.shape
        
        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 33  # approx 30fps
        
        if face_landmarks:
            for face in face_landmarks:
                # Capture reference landmarks on first detection
                if reference_landmarks is None:
                    reference_landmarks = face
                    cv2.putText(frame, "Reference Captured", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Compare with reference
                if reference_landmarks is not None:
                    dist = compute_face_distance(reference_landmarks, face)
                    if dist > thresh_dist:
                        person_change_flag += 1
                        if person_change_flag >= person_change_frame_check:
                            cv2.putText(frame, "DIFFERENT PERSON DETECTED!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            winsound.Beep(1200, 500)
                    else:
                        person_change_flag = 0
                
                rightEAR, rightPts = eye_aspect_ratio(RIGHT_EYE, face, img_w, img_h)
                leftEAR, leftPts = eye_aspect_ratio(LEFT_EYE, face, img_w, img_h)
                
                ear = (leftEAR + rightEAR) / 2.0
                
                # Draw contours around eyes
                cv2.polylines(frame, [np.array(rightPts, dtype=np.int32)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(leftPts, dtype=np.int32)], True, (0, 255, 0), 1)
                
                mar, mouthPts = mouth_aspect_ratio(MOUTH, face, img_w, img_h)
                
                # Draw mouth contour
                cv2.polylines(frame, [np.array(mouthPts, dtype=np.int32)], True, (255, 0, 0), 1)
                
                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "****************ALERT!****************", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        winsound.Beep(800, 500)  # Beep for drowsiness
                else:
                    flag = 0
                
                # Seizure alert only if both eye and mouth anomalies are detected
                if ear < thresh and mar > thresh_mar:
                    seizure_flag += 1
                    if seizure_flag >= seizure_frame_check:
                        cv2.putText(frame, "SEIZURE ALERT!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        winsound.Beep(1000, 500)  # Beep for seizure
                else:
                    seizure_flag = 0
                    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()
cap.release()
