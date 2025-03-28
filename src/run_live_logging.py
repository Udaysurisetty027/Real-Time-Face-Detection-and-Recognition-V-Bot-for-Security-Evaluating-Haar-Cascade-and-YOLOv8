import cv2
import face_recognition
from ultralytics import YOLO
from pickle_utils import load_encodings_pickle
from datetime import datetime
from logger import log_to_csv, should_log  # Import logging utilities

face_model = YOLO('yolov8n-face.pt')

encodings_path = "outputs/encodings.pkl"
known_face_encodings, known_face_names = load_encodings_pickle(encodings_path)

# Start video capture (0 for webcam, or replace with video file path)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = face_model(frame, conf=0.5)
    detected_faces = results[0].boxes.xyxy.cpu().numpy()

    recognized_faces = []
    for box in detected_faces:
        x_min, y_min, x_max, y_max = map(int, box)
        face_crop = frame[y_min:y_max, x_min:x_max]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(face_rgb)
        if encodings:
            face_encoding = encodings[0]

            # Match with known encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin() if matches else -1

            if matches and matches[best_match_index]:
                recognized_faces.append((box, known_face_names[best_match_index]))
            else:
                recognized_faces.append((box, "Unknown"))

    # Draw results on frame and log recognized faces
    frame_time = datetime.now()
    for (box, name) in recognized_faces:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Log recognized faces if criteria are met
        if should_log(name, frame_time):
            log_to_csv(name)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
