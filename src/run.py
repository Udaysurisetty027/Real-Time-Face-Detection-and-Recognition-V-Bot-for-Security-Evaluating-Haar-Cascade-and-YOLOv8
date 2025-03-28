from ultralytics import YOLO
import cv2
from pickle_utils import load_encodings_pickle
import face_recognition
import os

encodings_path = "outputs/encodings.pkl"

known_face_encodings, known_face_names = load_encodings_pickle(encodings_path)

if not known_face_encodings:
    print("No known face encodings found. Ensure the pickle file is updated.")

file_path = "datasets/sample.jpg"

face_model = YOLO("yolov8n-face.pt")

results = face_model(file_path, conf=0.5)
detected_faces = results[0].boxes.xyxy

image = cv2.imread(file_path)
face_crops = []

for box in detected_faces:
    x_min, y_min, x_max, y_max = map(int, box)
    crop = image[y_min:y_max, x_min:x_max]
    face_crops.append(crop)

recognized_faces = []

for face_crop in face_crops:
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(face_rgb)

    if encodings:
        face_encoding = encodings[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if matches:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                recognized_faces.append(known_face_names[best_match_index])
            else:
                recognized_faces.append("Unknown")
        else:
            recognized_faces.append("Unknown")
    else:
        recognized_faces.append("No Encoding Found")

for box, name in zip(detected_faces, recognized_faces):
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

output_path = "outputs/annotated_image.jpg"
cv2.imwrite(output_path, image)
print(f"Annotated image saved at {output_path}")
