import cv2
import numpy as np
import os
import pickle
from scipy.spatial.distance import cosine

# Paths to pretrained models
FACE_DETECTOR_PROTO = "Models/deploy.prototxt"
FACE_DETECTOR_MODEL = "Models/res10_300x300_ssd_iter_140000.caffemodel"
RECOGNITION_MODEL = "Models/nn4.small2.v1.t7"

# Load the face detector and recognizer models
face_net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_PROTO, FACE_DETECTOR_MODEL)
recog_net = cv2.dnn.readNetFromTorch(RECOGNITION_MODEL)

def get_face_embedding(face):
    """
    Generate face embeddings using the recognition model.
    """
    blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    recog_net.setInput(blob)
    embedding = recog_net.forward()
    return embedding.flatten()

def match_face(embedding, known_embeddings, threshold=0.3):
    """
    Match the input embedding with known embeddings.
    """
    min_distance = float('inf')
    label = "Unknown"
    for name, known_embedding in known_embeddings.items():
        distance = cosine(embedding, known_embedding)
        if distance < threshold and distance < min_distance:
            min_distance = distance
            label = name
    return label

def detect_faces(frame):
    """
    Detect faces in the frame using OpenCV DNN.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            faces.append((x1, y1, x2, y2))
    return faces

def draw_faces(frame, faces):
    """
    Draw rectangles around detected faces.
    """
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def generate_average_embedding_for_person(image_paths, name):
    """
    Generates average embedding for person.
    """
    embeddings = []
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        
        faces = detect_faces(image)
        
        if len(faces) > 0:
            (x1, y1, x2, y2) = faces[0]
            face = image[y1:y2, x1:x2]
            
            embedding = get_face_embedding(face)
            embeddings.append(embedding)
        else:
            print(f"No faces detected in {image_path}")
    
    if embeddings:
        average_embedding = np.mean(embeddings, axis=0)
        
        known_embeddings = load_known_embeddings()
        known_embeddings[name] = average_embedding
        
        save_known_embeddings(known_embeddings)
    else:
        print(f"No valid face embeddings found for {name}")


def load_known_embeddings():
    # Load the known embeddings from a file, if it exists
    if os.path.exists("known_embeddings.pkl"):
        with open("known_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    return {}

def save_known_embeddings(known_embeddings):
    # Save the known embeddings to a file
    with open("known_embeddings.pkl", "wb") as f:
        pickle.dump(known_embeddings, f)
