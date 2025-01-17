import cv2
from functions import get_face_embedding, match_face, detect_faces, draw_faces, load_known_embeddings

# Optional: Enable hardware acceleration
# face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Real-time face detection
cap = cv2.VideoCapture(0)  # Use your webcam (device 0)

while True:
    # Capture feed as frame, ret is boolean
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    faces = detect_faces(frame)
    
    # Draw bounding boxes
    draw_faces(frame, faces)

    # Load stored embeddings
    known_embeddings = load_known_embeddings() 

    # Match stored embeddings with embedding retrieved via webcam
    for (x1, y1, x2, y2) in faces:
        face = frame[y1:y2, x1:x2]
        embedding = get_face_embedding(face)
        label = match_face(embedding, known_embeddings)
        # Draw the label on the face
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the output
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
