import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from keras.utils import img_to_array
from collections import Counter

# Test the Trained Model with a Real-Time VideoCapture
#######################################################

STEPS = 6  # Frame-Steps for Signal
IMG_SHAPE = (480, 640, 3)
LM_SHAPE = (3, 21, 3)
CONFIDENCE = 0.8  # Confidence for Signal
LABELS = sorted(['C', 'Segunda', 'Casa', 'Obrigado', 'Rir', 'Aviao', 'Vazio'])
MODEL_PATH = './models/signals_model.h5'

cap = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils
hands = mp.solutions.hands
signal = hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=2)
mpose = mp.solutions.pose
pose = mpose.Pose()
HAND_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
HAND_CONNECTIONS_STYLE = mp.solutions.drawing_styles.get_default_hand_connections_style()
POSE_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
DRAW_STYLES = mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
model = load_model(MODEL_PATH)

predict = 0  # Start with no Prediction
while True:
    predicted_label = []
    for i in range(STEPS):  # STEPS Frames for Signal Prediction
        flag, frame = cap.read()
        if not flag:
            continue
        rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resulthand = signal.process(rgbframe)
        resultpose = pose.process(cv2.cvtColor(rgbframe, cv2.COLOR_BGR2RGB))
        if resulthand.multi_hand_landmarks and resultpose.pose_landmarks: # Only test when detects hand and pose
            if predict:
                cv2.putText(frame, "Signal: " + str(predict), (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, "Signal: " + str(predict), (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 220), 2, cv2.LINE_AA)
            image = np.zeros(frame.shape, dtype=np.uint8)  # Image for Test
            try:  # Try two hands in the frame
                hand1_landmarks = resulthand.multi_hand_landmarks[0]
                draw.draw_landmarks(image, hand1_landmarks, hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                draw.draw_landmarks(frame, hand1_landmarks, landmark_drawing_spec=DRAW_STYLES)
                hand2_landmarks = resulthand.multi_hand_landmarks[1]
                draw.draw_landmarks(image, hand2_landmarks, hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                draw.draw_landmarks(frame, hand2_landmarks, landmark_drawing_spec=DRAW_STYLES)
                pose_landmarks = resultpose.pose_landmarks
                draw.draw_landmarks(image, pose_landmarks, mpose.POSE_CONNECTIONS, POSE_LANDMARKS_STYLE)
                draw.draw_landmarks(frame, pose_landmarks, landmark_drawing_spec=DRAW_STYLES)
            except:
                try:  # Try one hand in the frame
                    hand1_landmarks = resulthand.multi_hand_landmarks[0]
                    draw.draw_landmarks(image, hand1_landmarks, hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                    draw.draw_landmarks(frame, hand1_landmarks, landmark_drawing_spec=DRAW_STYLES)
                    hand2_landmarks = False
                    pose_landmarks = resultpose.pose_landmarks
                    draw.draw_landmarks(image, pose_landmarks, mpose.POSE_CONNECTIONS, POSE_LANDMARKS_STYLE)
                    draw.draw_landmarks(frame, pose_landmarks, landmark_drawing_spec=DRAW_STYLES)
                except:
                    print('Error in Framming...')
                    break

            # Landmarks Coordinates for Test
            ps, lh, rh = np.zeros(LM_SHAPE[1:]), np.zeros(LM_SHAPE[1:]), np.zeros(LM_SHAPE[1:])
            if hand2_landmarks:
                if np.mean([h1lm.x for h1lm in hand1_landmarks.landmark]) <= np.mean([h2lm.x for h2lm in hand2_landmarks.landmark]):
                    for i in range(21):
                        ps[i][:] = [pose_landmarks.landmark[i].x, pose_landmarks.landmark[i].y, pose_landmarks.landmark[i].z]
                        lh[i][:] = [hand1_landmarks.landmark[i].x, hand1_landmarks.landmark[i].y, hand1_landmarks.landmark[i].z]
                        rh[i][:] = [hand2_landmarks.landmark[i].x, hand2_landmarks.landmark[i].y, hand2_landmarks.landmark[i].z]
                else:
                    for i in range(21):
                        ps[i][:] = [pose_landmarks.landmark[i].x, pose_landmarks.landmark[i].y, pose_landmarks.landmark[i].z]
                        rh[i][:] = [hand1_landmarks.landmark[i].x, hand1_landmarks.landmark[i].y, hand1_landmarks.landmark[i].z]
                        lh[i][:] = [hand2_landmarks.landmark[i].x, hand2_landmarks.landmark[i].y, hand2_landmarks.landmark[i].z]
            else:
                for i in range(21):
                    ps[i][:] = [pose_landmarks.landmark[i].x, pose_landmarks.landmark[i].y, pose_landmarks.landmark[i].z]
                    lh[i][:] = [hand1_landmarks.landmark[i].x, hand1_landmarks.landmark[i].y, hand1_landmarks.landmark[i].z]

            # Real-Time Test with Model Prediction
            test = [img_to_array(image).reshape((1,) + IMG_SHAPE), np.array([lh, rh, ps]).reshape((1,) + LM_SHAPE)]
            predicts = model.predict(test)
            print(np.argmax(predicts, axis=1))
            predicted_label.append(np.argmax(predicts, axis=1)[0])
        cv2.waitKey(100)
        cv2.imshow('Live', frame)

    # Test the confidence of predicted_label
    if len(predicted_label) > int(STEPS/2):
        pred = Counter(predicted_label).most_common(1)[0][0]
        if predicted_label.count(pred) > int(STEPS*CONFIDENCE):
            predict = LABELS[pred]
    else:
        predict = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
