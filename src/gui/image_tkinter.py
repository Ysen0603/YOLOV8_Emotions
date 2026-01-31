import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog

# Load YOLOv5 model
model = YOLO('./runs/classify/train/weights/best.pt')

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load image
img_path = 'mans happy face_24.jpeg'
img = cv2.imread(img_path)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

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
# Create Tkinter window
root = tk.Tk()
root.title("Results")
root.configure(bg='white')

# Convert image from BGR to RGB and resize 
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img1=cv2.imread(img_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1= cv2.resize(img1, (340, 340))
# Create canvas for displaying image
canvas = tk.Canvas(root,width=340, height=340, highlightthickness=0)
canvas.pack()

# Display image on canvas
photo = ImageTk.PhotoImage(image=Image.fromarray(img1))
canvas.create_image(0, 0, image=photo, anchor=tk.NW)

# Create frame for displaying results
results_frame = tk.Frame(root, bg='gray', padx=10, pady=10)
results_frame.config(width=400)
results_frame.pack()
#Loop over all detected faces
for (x, y, w, h) in faces:
    # Extract region of face
    face_img = img[y:y+h, x:x+w]

    # Predict class of object in face region using YOLOv5
    results = model(face_img)
    names_dict = results[0].names
    probs = results[0].probs.tolist()
    name_classes = names_dict[np.argmax(probs)]
    # Create label for displaying class name
    class_label = tk.Label(results_frame, text=name_classes, font=("Arial Black", 24), bg='#C0C0C0', fg='black', padx=10, pady=10, borderwidth=0, relief="groove", width=20)
    class_label.pack(side=tk.TOP, anchor=tk.W, fill=tk.X)
    

    
    # Display recommendations for emotion
    if name_classes in recommendations:
        rec_index = [0]
        rec_label = tk.Label(results_frame, text=recommendations[name_classes][rec_index[0]], font=("Arial", 16), bg='white', fg='black', padx=10, pady=10, borderwidth=0, relief="groove", width=40, wraplength=300)
        rec_label.pack(side=tk.TOP, anchor=tk.W, fill=tk.X)

        def next_recommendation(index):
            index[0] = (index[0] + 1) % len(recommendations[name_classes])
            rec_label.config(text=recommendations[name_classes][index[0]])

        next_button = tk.Button(results_frame, text="Recommandation suivante", font=("Arial", 18), bg='black', fg='white', padx=10, pady=5, borderwidth=2, relief="solid", command=lambda: next_recommendation(rec_index))
        next_button.pack(side=tk.TOP, anchor=tk.W, pady=10,fill=tk.X)

#Create button for closing window
close_button = tk.Button(root, text="Fermer la fenêtre", font=("Arial", 24), bg='#FF6666', fg='black', padx=10, pady=5, borderwidth=2, relief="solid", command=root.destroy)
close_button.pack(side=tk.BOTTOM, padx=10, pady=10)

#Run Tkinter main loop
root.mainloop()