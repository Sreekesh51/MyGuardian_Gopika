import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
from fer import FER
from collections import Counter

# --- File Paths ---
VIDEO_PATH = r"C:\Users\hp\Downloads\WhatsApp Video 2025-07-30 at 3.15.38 PM.mp4"
OUTPUT_CSV = "student_summary.csv"

# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_num = 0
logs = []

# --- Model Initialization ---
face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.6, max_num_hands=4)
emotion_detector = FER(mtcnn=True)

# --- Emotion Counter ---
emotion_counter = Counter()

# --- Frame Processing Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    timestamp = str(datetime.fromtimestamp(frame_num / fps))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection
    face_results = face_detector.process(rgb)
    head_count = len(face_results.detections) if face_results.detections else 0

    # Emotion Detection
    emotions = emotion_detector.detect_emotions(frame)
    top_emotions = []
    for face in emotions:
        emotion_scores = face["emotions"]
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        top_emotions.append(top_emotion)
        emotion_counter[top_emotion] += 1  # update running total

    emotion_summary = ", ".join(top_emotions) if top_emotions else "None"

    # Prepare Emotion Count Summary (running total string)
    emotion_count_str = ", ".join([f"{emotion}:{count}" for emotion, count in emotion_counter.items()])

    # Gesture Detection
    hand_results = hand_detector.process(rgb)
    gesture_type = "None"
    if hand_results.multi_hand_landmarks:
        hand_count = len(hand_results.multi_hand_landmarks)
        if hand_count >= 2:
            gesture_type = "Possible Fight"
        else:
            gesture_type = "Hand Raise"

    # Log Data
    logs.append({
        "Frame": frame_num,
        "Timestamp": timestamp,
        "Head Count": head_count,
        "Gesture": gesture_type,
        "Emotions": emotion_summary,
        "Emotion Count": emotion_count_str  # new column
    })

    # Overlay info
    cv2.putText(frame, f"Head Count: {head_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture_type}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if top_emotions:
        cv2.putText(frame, f"Emotions: {emotion_summary}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Student Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Save All Data to CSV ---
df = pd.DataFrame(logs)
df.to_csv(OUTPUT_CSV, index=False)

# --- Console Display ---
print(f"✅ Summary saved to {OUTPUT_CSV}")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()