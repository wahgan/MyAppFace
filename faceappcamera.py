import streamlit as st
import cv2
import numpy as np

# Load model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

# Load label map
label_map = np.load("label_map.npy", allow_pickle=True).item()

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("Webcam Face Recognition (LBPH)")
st.write("Press START to activate webcam. Press STOP to end.")

url=st.text_input("Masukan Streaming Alamat Kamera")
start = st.button("START")
stop = st.button("STOP")

frame_placeholder = st.empty()

# Session state to control camera
if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

cap = None

if st.session_state.run:
    cap = cv2.VideoCapture(url)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = model.predict(face)
            name = label_map.get(label, "Unknown")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({confidence:.1f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
