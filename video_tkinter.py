import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import datetime
import os

# Load YOLOv5 model
model = YOLO('./runs/classify/train/weights/best.pt')

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define recommendations for each emotion
recommendations = {
    'happy': 'Sortir avec des amis ou regarder un film comique',
    'sad': 'Parler a un ami ou un professionnel de la sante mentale',
    'angry': 'Faire des exercices physiques pour se defouler',
    'surprise': 'Prendre une pause pour reflechir à la situation',
    'disgust': 'Prendre une douche ou un bain pour se rafraichir',
    'fear': 'Pratiquer des exercices de respiration pour se calmer',
    'neutral': 'Faire une activite relaxante comme lire ou ecouter de la musique',
}

# Create Tkinter window
root = tk.Tk()
root.title("Détection d'émotions")
root.configure(bg='#F0F0F0')

# Create canvas for displaying image
canvas = tk.Canvas(root, width=540, height=340, highlightthickness=0, bg='#F0F0F0')
canvas.pack(pady=10)

# Create frame for displaying results
results_frame = tk.Frame(root, bg='#F0F0F0', padx=10, pady=10)
results_frame.pack(fill=tk.BOTH, expand=True)

# Create label for displaying class name
class_label = tk.Label(results_frame, text="", font=("Arial", 24, "bold"), bg='#4CAF50', fg='white', padx=10, pady=10, borderwidth=0, relief="flat")
class_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

# Create label for displaying recommendation
rec_label = tk.Label(results_frame, text="", font=("Arial", 16), bg='white', fg='black', padx=10, pady=10, borderwidth=1, relief="solid", wraplength=500)
rec_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))

# Create frame for control buttons
control_frame = tk.Frame(root, bg='#F0F0F0')
control_frame.pack(fill=tk.X, padx=10, pady=10)

# Global variables
cap = None
is_paused = False
is_webcam = False

def select_video():
    global cap, is_webcam, is_paused
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(file_path)
        is_webcam = False
        is_paused = False
        update_frame()  # Start updating frames immediately

def toggle_webcam():
    global cap, is_webcam, is_paused
    if cap is not None:
        cap.release()
    if is_webcam:
        is_webcam = False
        select_video()
    else:
        cap = cv2.VideoCapture(0)
        is_webcam = True
        is_paused = False
        update_frame()  # Start updating frames immediately

def play_pause():
    global is_paused
    is_paused = not is_paused
    if not is_paused:
        update_frame()

def forward():
    if cap is not None and not is_webcam:
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 30)  # Forward 30 frames
        update_frame()

def backward():
    if cap is not None and not is_webcam:
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 30))  # Backward 30 frames
        update_frame()

def capture_screenshot():
    if 'frame' in globals():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Screenshot saved as {filename}")

# Create styled buttons
style = ttk.Style()
style.configure('TButton', font=('Arial', 10), borderwidth=1)
style.map('TButton', background=[('active', '#3584e4')])

select_button = ttk.Button(control_frame, text="Sélectionner Vidéo", command=select_video, style='TButton')
select_button.pack(side=tk.LEFT, padx=5)

webcam_button = ttk.Button(control_frame, text="Basculer Webcam", command=toggle_webcam, style='TButton')
webcam_button.pack(side=tk.LEFT, padx=5)

play_pause_button = ttk.Button(control_frame, text="Lecture/Pause", command=play_pause, style='TButton')
play_pause_button.pack(side=tk.LEFT, padx=5)

forward_button = ttk.Button(control_frame, text="Avancer", command=forward, style='TButton')
forward_button.pack(side=tk.LEFT, padx=5)

backward_button = ttk.Button(control_frame, text="Reculer", command=backward, style='TButton')
backward_button.pack(side=tk.LEFT, padx=5)

screenshot_button = ttk.Button(control_frame, text="Capture d'écran", command=capture_screenshot, style='TButton')
screenshot_button.pack(side=tk.LEFT, padx=5)

close_button = ttk.Button(control_frame, text="Fermer", command=root.destroy, style='TButton')
close_button.pack(side=tk.RIGHT, padx=5)

def update_frame():
    global frame
    if cap is None:
        root.after(30, update_frame)
        return

    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            else:
                return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            results = model(face_img)
            names_dict = results[0].names
            
            top1_index = results[0].probs.top1
            name_classes = names_dict[top1_index]

            class_label.config(text=name_classes)

            if name_classes in recommendations:
                rec_label.config(text=recommendations[name_classes])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name_classes, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (540, 340))

        photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo

    root.after(30, update_frame)

# Run Tkinter main loop
root.mainloop()

# Release video file
if cap is not None:
    cap.release()