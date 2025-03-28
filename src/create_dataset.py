import cv2
import os
import time
import ttkbootstrap as tb
from tkinter import filedialog, messagebox

IMG_LIMIT = 30           # Number of images to capture
CAPTURE_INTERVAL = 0.5  
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
RESIZE_DIM = (200, 200)  

def create_face_dataset(username):
    if not username.strip():
        messagebox.showerror("Error", "Username cannot be empty.")
        return

    if not os.path.isfile(CASCADE_PATH):
        messagebox.showerror("Error", f"Haar Cascade file not found at {CASCADE_PATH}")
        return

    # creating user folder
    output_dir = f"datasets/{username}"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        messagebox.showerror("Error", f"Unable to create directory '{output_dir}'. {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    img_count = 0

    messagebox.showinfo("Info", f"Starting face capture for '{username}'. Press 'q' to quit.")

    try:
        while img_count < IMG_LIMIT:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Unable to read frame from the camera.")
                break

            small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                x, y, w, h = [int(coord * 2) for coord in (x, y, w, h)]
                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                face_resized = cv2.resize(face_img, RESIZE_DIM)
                img_count += 1
                img_path = os.path.join(output_dir, f"{img_count}.jpg")
                cv2.imwrite(img_path, face_resized)
                print(f"Saved: {img_path} ({img_count}/{IMG_LIMIT})")

                if img_count >= IMG_LIMIT:
                    break

            cv2.imshow('Face Dataset Creator', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting face capture...")
                break
            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Info", f"Face dataset creation completed. {img_count} images saved in '{output_dir}'.")

def start_gui():
    app = tb.Window(themename="superhero")
    app.title("Face Dataset Creator")
    app.geometry("500x400")

    header_label = tb.Label(app, text="Face Dataset Creator", font=("Helvetica", 20, "bold"), anchor="center")
    header_label.pack(pady=20)

    input_frame = tb.Frame(app)
    input_frame.pack(pady=20)

    username_label = tb.Label(input_frame, text="Enter Username:")
    username_label.pack(side="left", padx=10)

    username_entry = tb.Entry(input_frame, width=30)
    username_entry.pack(side="left", padx=10)

    preview_frame = tb.Frame(app, borderwidth=2, relief="groove")
    preview_frame.pack(pady=20, padx=20, fill="both", expand=True)

    preview_label = tb.Label(preview_frame, text="Camera Preview", anchor="center")
    preview_label.pack(pady=10)

    def on_start():
        username = username_entry.get()
        create_face_dataset(username)

    start_button = tb.Button(app, text="Start Capture", command=on_start, bootstyle="success")
    start_button.pack(pady=10)

    quit_button = tb.Button(app, text="Quit", command=app.destroy, bootstyle="danger")
    quit_button.pack(pady=10)

    app.mainloop()

if __name__ == "_main_":
    start_gui()