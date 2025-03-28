import pickle
import os

def load_encodings_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            data = pickle.load(file)

            if isinstance(data, tuple):
                return data  # Unpack directly

            if isinstance(data, dict):
                return data.get("encodings", []), data.get("names", [])

            raise ValueError("Unexpected data format in pickle file.")
    else:
        print(f"Pickle file not found at {file_path}. Returning empty encodings.")
        return [], []

def save_encodings_pickle(file_path, encodings, names):
    with open(file_path, "wb") as file:
        data = {"encodings": encodings, "names": names}
        pickle.dump(data, file)
        print(f"Encodings saved to {file_path}.")
