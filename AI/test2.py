import os
import cv2
import numpy as np
import pickle
from retinaface import RetinaFace
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# Paths
KNOWN_DIR = 'known_faces'
CACHE_FILE = 'embeddings_cache.pkl'
TEST_IMAGE = 'test.jpg'

# Initialize embedder
embedder = FaceNet()

# Step 1: Load or Generate Embeddings
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)
        known_names = data['names']
        known_embeddings = data['embeddings']
    print("[INFO] Loaded embeddings from cache.")
else:
    known_embeddings = []
    known_names = []
    print("[INFO] Generating embeddings...")

    for file in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, file)
        name = os.path.splitext(file)[0]

        img = cv2.imread(path)
        faces = RetinaFace.detect_faces(img)
        if not faces:
            continue

        face = list(faces.values())[0]
        x1, y1, x2, y2 = face['facial_area']
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (160, 160))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        embedding = embedder.embeddings([resized])[0]
        known_embeddings.append(embedding)
        known_names.append(name)

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'names': known_names, 'embeddings': known_embeddings}, f)
    print("[INFO] Saved embeddings to cache.")

# Step 2: Recognize Person in New Image
img = cv2.imread(TEST_IMAGE)
detected_faces = RetinaFace.detect_faces(img)

for key in detected_faces:
    face = detected_faces[key]
    x1, y1, x2, y2 = face['facial_area']
    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (160, 160))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    test_embedding = embedder.embeddings([resized])[0]

    # Compare with cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cosine_similarity(test_embedding, e) for e in known_embeddings]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    name = "Unknown"
    if best_score > 0.7:
        name = known_names[best_idx]

    # Draw on image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show result
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Face Recognition Result")
plt.show()
