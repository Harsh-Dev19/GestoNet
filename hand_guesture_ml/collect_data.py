import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

data = []
label = input("Enter Gesture Name: ")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, f"Gesture: {label}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                data.append(landmarks + [label])
                print("Saved")

    cv2.imshow("Collect Data - Press S", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

columns = [f"f{i}" for i in range(63)] + ["label"]
df = pd.DataFrame(data, columns=columns)

if os.path.exists("gestures.csv"):
    df.to_csv("gestures.csv", mode='a', header=False, index=False)
else:
    df.to_csv("gestures.csv", index=False)

print("Data collection complete!")
