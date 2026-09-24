<div align="center">

# ◈ GestoNet

### COMPUTER VISION • MACHINE LEARNING • HAND GESTURE RECOGNITION

**A real-time hand gesture recognition system built with Python.**

<br>

<img src="https://img.shields.io/badge/PYTHON-0D0D0D?style=for-the-badge&logo=python&logoColor=9FE3C1" />
<img src="https://img.shields.io/badge/OPENCV-0D0D0D?style=for-the-badge&logo=opencv&logoColor=C77D8A" />
<img src="https://img.shields.io/badge/MEDIAPIPE-0D0D0D?style=for-the-badge&logo=google&logoColor=9FE3C1" />
<img src="https://img.shields.io/badge/SCIKIT--LEARN-0D0D0D?style=for-the-badge&logo=scikit-learn&logoColor=C77D8A" />

<br><br>

</div>

---

# ◈ OVERVIEW

**GestoNet** is a real-time hand gesture recognition system that uses a webcam, MediaPipe hand landmarks, and a machine learning classifier to recognize custom hand gestures.

The system converts a detected hand into a numerical feature vector containing **63 landmark coordinates**, trains a **Random Forest classifier** on collected gesture samples, and uses the trained model to recognize gestures from live webcam input.

The complete workflow is:

`WEBCAM` → `HAND LANDMARKS` → `FEATURES` → `TRAINING` → `MODEL` → `REAL-TIME PREDICTION`

---

# ◈ FEATURES

### 🖐️ Real-Time Hand Detection

The system uses MediaPipe Hands to detect a single hand through the webcam.

The detected hand contains **21 landmarks**, with each landmark providing:

- X coordinate
- Y coordinate
- Z coordinate

This produces:

`21 × 3 = 63 FEATURES`

### 📊 Custom Gesture Dataset

Gestures can be collected directly through the webcam.

The user provides a gesture name and can save landmark samples while performing that gesture.

The collected data is stored in:

~~~text
gestures.csv
~~~

### 🧠 Machine Learning Classification

GestoNet uses a **Random Forest Classifier** to learn patterns from the collected landmark data.

The dataset is divided into training and testing sets before training.

### 🎯 Real-Time Prediction

The trained model can be loaded from:

~~~text
model.pkl
~~~

and used to classify gestures from live webcam frames.

### 🦴 Landmark Visualization

MediaPipe hand connections are drawn directly over the detected hand, making the recognition process visible while the system runs.

### 🔁 Customizable Gestures

New gesture classes can be added by collecting additional samples with different gesture labels and retraining the model.

---

# ◈ HOW IT WORKS

~~~text
                 WEBCAM
                    │
                    ▼
             OPENCV FRAME
                    │
                    ▼
             MEDIAPIPE HAND
                DETECTION
                    │
                    ▼
             21 HAND LANDMARKS
                    │
                    ▼
          X / Y / Z COORDINATES
                    │
                    ▼
             63 FEATURES
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     DATA COLLECTION      LIVE INPUT
          │                   │
          ▼                   ▼
    gestures.csv         TRAINED MODEL
          │                   │
          ▼                   ▼
   RANDOM FOREST ──────► PREDICTION
          │                   │
          ▼                   ▼
      model.pkl         GESTURE LABEL
~~~

---

# ◈ DATA COLLECTION

Gesture samples are collected using:

~~~text
collect_data.py
~~~

When the script starts, it asks for a gesture name.

For example:

~~~text
Enter Gesture Name: thumbs_up
~~~

The webcam then detects the hand and extracts its 63 landmark values.

Press:

~~~text
S → Save current gesture sample
ESC → Stop collection
~~~

The samples are appended to `gestures.csv`.

The dataset follows this structure:

~~~text
f0, f1, f2, ... f61, f62, label
~~~

The first 63 columns represent the hand landmark features, while the final column contains the gesture label.

---

# ◈ MODEL TRAINING

Training is handled by:

~~~text
train_model.py
~~~

The script:

1. Loads `gestures.csv`
2. Separates features and labels
3. Splits the dataset into training and testing sets
4. Creates a Random Forest classifier
5. Trains the classifier
6. Evaluates the model
7. Saves the trained model as `model.pkl`

The dataset uses an **80/20 train-test split**.

The Random Forest model is configured with:

~~~text
n_estimators = 200
~~~

The resulting accuracy is printed to the console after training.

---

# ◈ REAL-TIME PREDICTION

Prediction is handled by:

~~~text
predict.py
~~~

The script loads the trained model:

~~~python
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
~~~

It then continuously:

1. Captures webcam frames
2. Flips the frame horizontally
3. Converts the frame to RGB
4. Detects hand landmarks using MediaPipe
5. Extracts the 63 landmark features
6. Sends the features to the trained Random Forest model
7. Displays the predicted gesture on the webcam feed

If no hand is detected, the system displays:

~~~text
No Hand
~~~

Press `ESC` to exit the prediction window.

---

# ◈ TECHNOLOGY

### PROGRAMMING LANGUAGE

`Python`

### COMPUTER VISION

`OpenCV`

### HAND LANDMARK DETECTION

`MediaPipe Hands`

### MACHINE LEARNING

`Scikit-learn`

### CLASSIFICATION MODEL

`Random Forest Classifier`

### DATA PROCESSING

`Pandas` · `NumPy`

### MODEL STORAGE

`Pickle`

---

# ◈ PROJECT STRUCTURE

~~~text
GestoNet/
│
└── hand_guesture_ml/
    │
    ├── collect_data.py
    ├── train_model.py
    ├── predict.py
    ├── gestures.csv
    └── model.pkl
~~~

### `collect_data.py`

Collects hand landmark samples from the webcam and stores them in the gesture dataset.

### `train_model.py`

Loads the dataset, trains the Random Forest classifier, evaluates it, and saves the trained model.

### `predict.py`

Loads the trained model and performs real-time gesture recognition through the webcam.

### `gestures.csv`

Contains the collected 63-feature hand landmark dataset and corresponding gesture labels.

### `model.pkl`

Stores the trained Random Forest model used during real-time prediction.

---

# ◈ SETUP

### 1. Clone the repository

~~~bash
git clone https://github.com/Harsh-Dev19/GestoNet.git
cd GestoNet/hand_guesture_ml
~~~

### 2. Install dependencies

~~~bash
pip install opencv-python mediapipe pandas numpy scikit-learn
~~~

### 3. Make sure a webcam is available

GestoNet uses the system's default webcam for both data collection and prediction.

---

# ◈ TRAIN YOUR OWN GESTURES

### Step 1 — Collect samples

Run:

~~~bash
python collect_data.py
~~~

Enter the name of the gesture you want to collect.

Example:

~~~text
Enter Gesture Name: thumbs_up
~~~

Perform the gesture in front of the webcam and press `S` to save samples.

Press `ESC` when finished.

Repeat the process for additional gesture classes.

---

### Step 2 — Train the model

After collecting your gesture samples, run:

~~~bash
python train_model.py
~~~

The script will train the Random Forest classifier and generate:

~~~text
model.pkl
~~~

---

### Step 3 — Run recognition

Start real-time prediction with:

~~~bash
python predict.py
~~~

A webcam window will open and the predicted gesture will appear on screen.

Press `ESC` to stop.

---

# ◈ GESTURE PIPELINE

~~~text
COLLECT
   │
   ▼
21 HAND LANDMARKS
   │
   ▼
63 NUMERICAL FEATURES
   │
   ▼
CSV DATASET
   │
   ▼
TRAIN / TEST SPLIT
   │
   ▼
RANDOM FOREST
   │
   ▼
model.pkl
   │
   ▼
LIVE WEBCAM
   │
   ▼
GESTURE PREDICTION
~~~

---

# ◈ PROJECT HIGHLIGHTS

- Real-time webcam-based recognition
- MediaPipe hand landmark detection
- 21-point hand representation
- 63 numerical landmark features
- Custom gesture dataset collection
- Random Forest classification
- Train/test evaluation
- Saved trained model
- Real-time prediction
- Visual hand landmark tracking
- Custom gesture support

---

# ◈ CURRENT SCOPE

GestoNet currently focuses on recognizing custom hand gestures from a single detected hand.

The system uses hand landmark coordinates as machine learning features rather than directly training on raw image pixels.

The current implementation supports:

- One hand at a time
- Webcam input
- Custom gesture labels
- Random Forest classification
- Real-time prediction

---

# ◈ FUTURE DIRECTIONS

Possible extensions include:

- Multi-hand gesture recognition
- Larger gesture datasets
- More gesture classes
- Improved feature normalization
- Gesture confidence scores
- Dynamic gesture recognition
- Gesture-controlled applications
- Real-time gesture command mapping
- Expanded model evaluation

---

<div align="center">

### ◈ GestoNet

**SEE → EXTRACT → LEARN → RECOGNIZE**

<br>

`PYTHON` · `OPENCV` · `MEDIAPIPE` · `MACHINE LEARNING`

<br><br>

**Harsh Dev**

</div>
