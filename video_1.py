import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

# Load YOLOv5 model
model = YOLO('./runs/classify/train/weights/best.pt')

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define recommendations for each emotion
recommendations = {
    'happy': ['Sortir avec des amis ou regarder un film comique', 'Faire du sport ou de l\'exercice', 'Ecouter de la musique joyeuse', 'Manger un aliment que vous aimez'],
    'sad': ['Parler a un ami ou un professionnel de la sante mentale', 'Faire une activite creative comme dessiner ou peindre', 'Faire une promenade en plein air', 'Regarder un film triste pour liberer vos emotions'],
    'angry': ['Faire des exercices physiques pour se defouler', 'Ecrire dans un journal pour exprimer vos emotions', 'Pratiquer la meditation ou la relaxation', 'Ecouter de la musique calme'],
    'surprise': ['Prendre une pause pour reflechir à la situation', 'Faire une activite que vous n\'avez jamais essayee auparavant', 'Parler a un ami ou un professionnel de la sante mentale', 'Faire une liste des avantages et des inconvenients de la situation'],
    'disgust': ['Prendre une douche ou un bain pour se rafraichir', 'Faire une activite que vous aimez pour vous changer les idees', 'Parler a un ami ou un professionnel de la sante mentale', 'Faire une liste des choses pour lesquelles vous etes reconnaissant'],
    'fear': ['Pratiquer des exercices de respiration pour se calmer', 'Faire une activite relaxante comme lire ou ecouter de la musique', 'Parler a un ami ou un professionnel de la sante mentale', 'Faire une liste des choses que vous pouvez faire pour vous preparer à la situation'],
    'neutral': ['Faire une activite relaxante comme lire ou ecouter de la musique', 'Faire une activite que vous aimez pour vous changer les idees', 'Parler a un ami ou un professionnel de la sante mentale', 'Faire une liste des choses pour lesquelles vous etes reconnaissant'],
}
# Add variable to track video state
paused = False
# Create Tkinter window
root = tk.Tk()
root.title("Results")
root.configure(bg='white')

# Create canvas for displaying image
canvas = tk.Canvas(root,width=640, height=340, highlightthickness=0)
canvas.pack()

# Create frame for displaying results
results_frame = tk.Frame(root, bg='gray', padx=10, pady=10)
results_frame.config(width=400)
results_frame.pack()

# Create label for displaying class name
class_label = tk.Label(results_frame, text="", font=("Arial Black", 24), bg='#C0C0C0', fg='black', padx=10, pady=10, borderwidth=0, relief="groove", width=20)
class_label.pack(side=tk.TOP, anchor=tk.W, fill=tk.X)

# Create label for displaying recommendations
rec_label = tk.Label(results_frame, text="", font=("Arial", 16), bg='white', fg='black', padx=10, pady=10, borderwidth=0, relief="groove", width=40, wraplength=300)
rec_label.pack(side=tk.TOP, anchor=tk.W, fill=tk.X)

# Create button for displaying next recommendation
rec_index = [0]
next_button = tk.Button(results_frame, text="Recommandation suivante", font=("Arial", 18), bg='black', fg='white', padx=10, pady=5, borderwidth=2, relief="solid", command=lambda: next_recommendation(rec_index))
next_button.pack(side=tk.TOP, anchor=tk.W, pady=10,fill=tk.X)

# Create button for pausing video
pause_button = tk.Button(root, text="Pause", font=("Arial", 24), bg='#FF6666', fg='black', padx=10, pady=5, borderwidth=2, relief="solid", command=lambda: pause_video(cap))
pause_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create button for resuming video
resume_button = tk.Button(root, text="Reprendre", font=("Arial", 24), bg='#66FF66', fg='black', padx=10, pady=5, borderwidth=2, relief="solid", command=lambda: resume_video(cap))
resume_button.pack(side=tk.LEFT, padx=10, pady=10)

# Function to pause video
def pause_video(cap):
    global paused
    if cap.isOpened():
        paused = True
        cap.release()

# Function to resume video
# Function to resume video
def resume_video(cap):
    global paused
    if not cap.isOpened():
        print("Video file is not open")
        if cap.open(video_path):
            print("Video file opened successfully")
        else:
            print("Error opening video file")
            return
    paused = False




#Create button for closing window
close_button = tk.Button(root, text="Fermer la fenêtre", font=("Arial", 24), bg='#FF6666', fg='black', padx=10, pady=5, borderwidth=2, relief="solid", command=root.destroy)
close_button.pack(side=tk.RIGHT, padx=10, pady=10)
#Define function for displaying next recommendation
def next_recommendation(index):
    index[0] = (index[0] + 1) % len(recommendations[class_label['text']])
    rec_label.config(text=recommendations[class_label['text']][index[0]])


#Open video file
video_path = 'video2.mp4'
cap = cv2.VideoCapture(video_path)

#Loop over all frames in video
while True:
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
        probs = results[0].probs.tolist()
        name_classes = names_dict[np.argmax(probs)]

        # Update class label
        class_label.config(text=name_classes)

        # Update recommendations label
        if name_classes in recommendations:
            rec_index[0] = 0
            rec_label.config(text=recommendations[name_classes][rec_index[0]])

    # Convert frame from BGR to RGB and resize 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 340))

    # Display frame on canvas
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    # Update Tkinter window
    root.update()
#Release video file
cap.release()
#Run Tkinter main loop
root.mainloop()


