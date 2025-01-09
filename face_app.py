import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import glob
import subprocess
import sys

# Directory Paths
facedir = 'data'
traindir = 'train_face'
os.makedirs(facedir, exist_ok=True)
os.makedirs(traindir, exist_ok=True)

# Face Detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit App Title
st.title("Face Recognition Security System")

# Sidebar for Navigation
mode = st.sidebar.radio("Select Mode", ["Face Recording", "Face Training", "Face Recognition"])
camera = cv2.VideoCapture(1)

# --- FACE RECORDING ---
if mode == "Face Recording":
    st.header("Face Recording")
    face_id = st.text_input("Enter Face ID:", "")

    if st.button("Start Recording"):
        if not face_id:
            st.warning("Please enter a valid Face ID.")
        else:
            # Initialize webcam
            #camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                st.error("Webcam not detected. Please check your camera settings.")
                st.stop()

            camera.set(3, 640)
            camera.set(4, 480)
            count = 1
            st.write("Capturing faces... Press 'Stop' or close the window when done.")
            FRAME_WINDOW = st.image([])

            try:
                while count <= 300:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Failed to access the webcam.")
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        face_img = gray[y:y + h, x:x + w]
                        filename = f'{facedir}/face.{face_id}.{count}.jpg'
                        cv2.imwrite(filename, face_img)
                        count += 1
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    FRAME_WINDOW.image(frame, channels="BGR")

            finally:
                camera.release()
                cv2.destroyAllWindows()

            st.success(f"Face recording completed. {count - 1} images saved.")

# --- FACE TRAINING ---
elif mode == "Face Training":
    st.header("Face Training")
    if st.button("Train Model"):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Function to Get Image Label
        def get_image_label(path):
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('jpg', 'jpeg', 'png'))]
            face_samples = []
            face_ids = []

            for image_path in image_paths:
                PILImg = Image.open(image_path).convert('L')
                img_numpy = np.array(PILImg, 'uint8')

                try:
                    face_id = int(os.path.split(image_path)[-1].split(".")[1])
                except ValueError:
                    st.warning(f"Skipping invalid file: {image_path}")
                    continue

                faces = face_detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    face_ids.append(face_id)

            return face_samples, face_ids

        st.write("Training in progress...")
        faces, ids = get_image_label(facedir)

        if faces:
            face_recognizer.train(faces, np.array(ids))
            face_recognizer.write(os.path.join(traindir, 'training.xml'))
            st.success(f"Training completed on {len(np.unique(ids))} unique faces.")
        else:
            st.error("No valid training data found.")

# --- FACE RECOGNITION ---
elif mode == "Face Recognition":
    st.title("Face Recognition for Authentication")
    st.write("Start the camera to authenticate.")

    start_button = st.button("Start Face Recognition")

    # Initialize the face recognizer and check if trained model exists
    model_path = os.path.join(traindir, 'training.xml')
    if os.path.exists(model_path):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(model_path)
    else:
        st.error("Training model not found. Please train the model first.")
        st.stop()

    if start_button:
        stframe = st.empty()
        access_granted = False

        # Initialize webcam
        #cam = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Webcam not detected. Please check your camera settings.")
            st.stop()

        camera.set(3, 640)
        camera.set(4, 480)

        # Define font for text display
        font = cv2.FONT_HERSHEY_SIMPLEX

        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to access the webcam.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    id, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])
                    if confidence <= 50:
                        name = "owner"  # Adjust this to match your owner's name or ID
                        confidence_text = f"{round(100 - confidence)}%"
                        access_granted = True
                    else:
                        name = "unknown"
                        confidence_text = f"{round(100 - confidence)}%"

                    # Display the name and confidence on the frame
                    cv2.putText(frame, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                    cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                stframe.image(frame, channels="BGR")

                if access_granted:
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

        if access_granted:
            st.success("Access Granted! Welcome, owner.")
            folder_path = r"C:\Users\asus\Desktop\UKM SEM 5\TC3413(ROBOT)\face_app\with_mask"
            st.write(f"Protected folder: {folder_path}")

            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
                if not files:
                    st.warning("No files found in the folder.")
                else:
                    selected_file = st.selectbox("Select a file to open:", files)
                
                if st.button("Open Selected File"):
                    file_path = os.path.join(folder_path, selected_file)
                    print(file_path, selected_file)
                    st.write(f"Selected File Path: {file_path}")

                    if os.path.isfile(file_path):
                        st.write("Attempting to open the file...")
                        try:
                            print("try")
                            if os.name == 'nt':  # Windows
                                print("start")
                                os.startfile(file_path)
                            elif sys.platform == 'darwin':  # macOS
                                subprocess.Popen(['open', file_path])
                            else:  # Linux
                                subprocess.Popen(['xdg-open', file_path])
                            st.success(f"File opened successfully: {selected_file}")
                        except Exception as e:
                            st.error(f"Error opening file: {e}")
                    else:
                        print("file x wujud")
                        st.error("The selected file does not exist or is invalid.")
            else:
                print("x jumpa path")
                st.error("The specified folder path does not exist.")
