import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv5 model
model = YOLO('./runs/classify/train/weights/best.pt')

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define recommendations for each emotion
recommendations = {
    'happy': ['Sortir avec des amis ou regarder un film comique'],
    'sad': ['Parler a un ami ou un professionnel de la sante mentale'],
    'angry': ['Faire des exercices physiques pour se defouler'],
    'surprise': ['Prendre une pause pour reflechir à la situation'],
    'disgust': ['Prendre une douche ou un bain pour se rafraichir'],
    'fear': ['Pratiquer des exercices de respiration pour se calmer'],
    'neutral': ['Faire une activite relaxante comme lire ou ecouter de la musique'],
}

# Open video file
cap = cv2.VideoCapture('video2.mp4')
#cap = cv2.VideoCapture('http://192.168.1.4:8080/video')

# Loop over all frames in the video
while cap.isOpened():
    # Read frame from video
    ret, frame = cap.read()

    # Stop if end of video
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Extract region of face
        face_img = frame[y:y+h, x:x+w]

        # Predict class of object in face region using YOLOv5
        results = model(face_img)
        names_dict = results[0].names
        
        # Use top1 index to get the most likely class
        top1_index = results[0].probs.top1
        name_classes = names_dict[top1_index]

        # Draw rectangle around face and display class name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), 2)
        cv2.putText(frame, name_classes, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 250), 2)

        # Display recommendations for emotion
        if name_classes in recommendations:
            for rec in recommendations[name_classes]:
                # Get size of recommendation text
                (rec_w, rec_h), _ = cv2.getTextSize(rec, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Calculate position of recommendation text
                rec_x = x + int((w - rec_w) / 2)
                rec_y = y + h + 30

                # Draw recommendation text
                cv2.putText(frame, rec, (rec_x, rec_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 250), 2)

                # Update y position for next recommendation
                y += rec_h + 10

    # Display frame
    cv2.imshow('Video', frame)

    # Exit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()