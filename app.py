import cv2
import torch
import tkinter as tk
from tkinter import simpledialog
from torchvision.transforms import ToTensor

from functions.app_design import app_theme
from functions.model_emotion_classifier import EmotionClassifier


# Create a tkinter root window
root = tk.Tk()
root.withdraw()
app_theme(root)

# Ask the user for the camera number
camera_number = None
while camera_number is None:
    dialogue = simpledialog.askstring("Input", "Please enter your camera number (if you only have 1, then it's 0 because indexing starts from 0)", parent=root)
    if dialogue.isdigit():
        camera_number = int(dialogue)

# Create a dictionary to map class numbers to emotion names
class_to_emotion = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create an instance of the model & Move model to GPU if available
model = EmotionClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load model weights & # Make sure the model is in evaluation mode
model.load_state_dict(torch.load('weights/best_emotion_classifier.pth'))
model.eval()

# Load cascading classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv2.VideoCapture(camera_number)

while True:
    ret, frame = cap.read() # Read image from camera
    if not ret:
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no face is detected, show the image and continue the loop
    if len(faces) == 0:
        cv2.imshow('Emotion Detection (space to exit)', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
        continue

    # Otherwise, select the largest face
    face = max(faces, key=lambda rectangle: rectangle[2] * rectangle[3])

    # Extract face coordinates
    x, y, w, h = face

    # Extract face from image & convert to tensor
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    image = ToTensor()(face).unsqueeze(0)

    # If you are using a GPU, move the image to the GPU
    image = image.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1)

    # Get the name of the predicted emotion
    predicted_emotion = class_to_emotion[predicted_class.item()]

    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show predicted emotion above rectangle
    cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show image
    cv2.imshow('Emotion Detection (space to exit)', frame)

    # If user presses 'space', exit loop
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Free the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()