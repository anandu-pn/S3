```markdown
# 🧠 Face Recognition System Using RetinaFace + FaceNet

This project implements a **face recognition system** that uses:

- ✅ **RetinaFace** for face detection
- ✅ **FaceNet (via keras-facenet)** for face embedding and comparison

It compares an unknown face (from a test image) with known faces stored in a folder and identifies the person using cosine similarity.

---

## 📁 Project Structure
```

face_recognition_project/
├── known_faces/ # Folder with known person images
│ ├── alice.jpg
│ ├── bob.jpg
├── test.jpg # Test image to identify
├── recognize_faces.py # Main script
├── embeddings_cache.pkl # Auto-generated cache for speed
└── README.md # This file

````

---

## ⚙️ Environment Setup

### ✅ 1. Create Conda Environment

```bash
conda create -n face_env python=3.9 -y
conda activate face_env
````

---

### ✅ 2. Install Dependencies

```bash
pip install retina-face keras-facenet opencv-python numpy matplotlib scipy
```

#### 🧩 Optional Fix for TensorFlow >= 2.17.1

If you get an error like:

> `ValueError: You have tensorflow X.X.X and this requires tf-keras package.`

Run:

```bash
pip install tf-keras
```

---

## 🚀 How It Works

1. All images in the `known_faces/` folder are processed and encoded using FaceNet.
2. Face embeddings are cached in `embeddings_cache.pkl` (only computed once).
3. The test image (`test.jpg`) is analyzed:

   - Face detected with RetinaFace
   - Embedding extracted with FaceNet
   - Compared with known embeddings using cosine similarity

4. Identity is displayed on the image.

---

## ▶️ Run the Program

For run without caching

```bash
python recognize_faces_1.py
```

For run with catching
For run without caching

```bash
python recognize_faces_2.py
```

If it's your first time running the script, it will generate and save face embeddings for known faces.

Subsequent runs are much faster due to caching.

---

## 🔁 Add or Remove Faces

To update the database:

1. Add/remove images inside the `known_faces/` directory.
2. Delete the cache:

```bash
rm embeddings_cache.pkl
```

3. Run the script again:

```bash
python recognize_faces.py
```

---

## 🧠 Recognition Logic

- The face similarity is computed using **cosine similarity**.
- A threshold (`> 0.7`) is used to decide whether it is a known person.
- You can tweak this threshold for higher or lower tolerance.

---

## 🧪 Example Output

After processing, the image will be displayed with bounding boxes and predicted names:

```
[INFO] Recognized: Bob (0.84)
```

_(If using matplotlib, the image will open in a window)_

---

## 🔧 Optional Improvements

| Feature          | Description                                |
| ---------------- | ------------------------------------------ |
| Real-time webcam | Integrate OpenCV to recognize via webcam   |
| GUI or Web app   | Use Tkinter or Flask for UI                |
| Attendance log   | Save identified names and timestamps       |
| REST API         | Deploy as backend face recognition service |
| Training         | Use your own FaceNet weights (advanced)    |

---

## 🧾 Dependencies

| Package         | Purpose                          |
| --------------- | -------------------------------- |
| `retina-face`   | Face detection (RetinaFace)      |
| `keras-facenet` | Face embeddings (FaceNet)        |
| `opencv-python` | Image processing & visualization |
| `numpy`         | Numerical computation            |
| `scipy`         | Cosine similarity calculations   |
| `matplotlib`    | Display image with results       |
| `tf-keras`      | Optional Keras compatibility     |

---

## 🛡 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
Free for academic, personal, and non-commercial use.

---

## 🙋‍♂️ Need Help?

- Make sure all images are **clear frontal face photos**.
- Use `python=3.9` as it's widely compatible with TensorFlow 2.x.
- If you're stuck, open an issue or reach out.

Happy coding! 🚀

```

```
