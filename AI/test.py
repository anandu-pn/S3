import os
import cv2
import numpy as np
from retinaface import RetinaFace
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# Initialize FaceNet model
embedder = FaceNet()

# Folder with known faces
known_folder = 'known_faces'
known_embeddings = []
known_names = []

# Step 1: Generate embeddings for known faces
for filename in os.listdir(known_folder):
    path = os.path.join(known_folder, filename)
    name = os.path.splitext(filename)[0]

    img = cv2.imread(path)
    faces = RetinaFace.detect_faces(img)

    if not faces:
        continue

    # Use the first detected face
    face = list(faces.values())[0]
    x1, y1, x2, y2 = face['facial_area']
    cropped_face = img[y1:y2, x1:x2]

    # Resize to 160x160 as expected by FaceNet
    resized = cv2.resize(cropped_face, (160, 160))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Get embedding
    embedding = embedder.embeddings([resized])[0]

    known_embeddings.append(embedding)
    known_names.append(name)

print("[INFO] Known faces loaded")

# Step 2: Recognize unknown face
test_img = cv2.imread("test.jpg")
test_faces = RetinaFace.detect_faces(test_img)

for key in test_faces:
    face = test_faces[key]
    x1, y1, x2, y2 = face['facial_area']
    cropped = test_img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (160, 160))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    test_embedding = embedder.embeddings([resized])[0]

    # Compare with known embeddings using cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_similarity(test_embedding, emb) for emb in known_embeddings]
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[best_match_index]

    name = "Unknown"
    if best_similarity > 0.7:
        name = known_names[best_match_index]

    # Draw result
    cv2.rectangle(test_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(test_img, name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Show final result
img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Face Recognition Result")
plt.show()
