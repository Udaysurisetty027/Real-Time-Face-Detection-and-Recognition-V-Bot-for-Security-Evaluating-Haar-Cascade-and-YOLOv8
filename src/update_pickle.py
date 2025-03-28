import os
import face_recognition
from pickle_utils import save_encodings_pickle

known_faces_path = "datasets/known_faces"
encodings_path = "outputs/encodings.pkl"

all_face_encodings = []
all_face_names = []

# known faces folder
for person_name in os.listdir(known_faces_path):
    person_folder = os.path.join(known_faces_path, person_name)

    if os.path.isdir(person_folder):
        for image_file in os.listdir(person_folder):
            if image_file.endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_folder, image_file)

                # Load and encode the image
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    all_face_encodings.append(encodings[0])
                    all_face_names.append(person_name)

#pickle file
save_encodings_pickle(encodings_path, all_face_encodings, all_face_names)
print("Encoding process completed.")
