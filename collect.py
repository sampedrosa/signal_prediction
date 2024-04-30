import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from keras.utils import img_to_array

# Collect an Image of the Landmarks (jpg) and their coordinates xyz (pkl)
#########################################################################

STEPS = 3   # Frame-Steps to use the Mean of STEPS frames
PRINT = 70  # Maximum Frames for each Print-Collect
IMAGE_SHAPE = (480, 640, 3)
LM_SHAPE = (3, 21, 3)
LABELS = ['C', 'Segunda', 'Casa', 'Obrigado', 'Rir', 'Aviao', 'Vazio']  # Signals to Collect

# Collect/Create Data for a list of Labels (Signals)
def collect_data(dir, labels):
    # Define VideoCapture and Mediepipe (Hands and pose) Variables
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils
    hands = mp.solutions.hands
    signal = hands.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=2)
    mpose = mp.solutions.pose
    pose = mpose.Pose()
    HAND_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
    HAND_CONNECTIONS_STYLE = mp.solutions.drawing_styles.get_default_hand_connections_style()
    POSE_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    for label in labels:
        print(f'Collection of Label: {label}')
        errors, count = 0, 0
        if not os.path.exists(dir + '/images/' + label):
            os.makedirs(dir + '/images/' + label)
            os.makedirs(dir + '/landmarks/' + label)
        else:
            count = len(os.listdir(dir + '/images/' + label))

        while True:
            flag, frame = cap.read()
            if not flag:
                continue
            cv2.putText(frame, label + ': S for Start, N for Next', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(frame, label + ': S for Start, N for Next', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 220), 2, cv2.LINE_AA)
            cv2.imshow('Live', frame)
            key = cv2.waitKey(50)
            if key == ord('n'):  # Press N: Finish collecting actual Label and proceed to Next Label (Signal)
                break
            if key == ord('s'):  # Press S: Start collecting (Print)
                cv2.waitKey(100)
                for i in range(PRINT):
                    flag, frame = cap.read()
                    if not flag:
                        continue
                    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resulthand = signal.process(rgbframe)
                    resultpose = pose.process(cv2.cvtColor(rgbframe, cv2.COLOR_BGR2RGB))
                    if resulthand.multi_hand_landmarks and resultpose.pose_landmarks:  # Only collects when detects hand and pose
                        image = np.zeros(frame.shape, dtype=np.uint8)
                        live = np.zeros((frame.shape[0], frame.shape[1] * 2, frame.shape[2]), dtype=np.uint8)
                        hand1_landmarks, hand2_landmarks, pose_landmarks, images = [], [], [], []
                        for j in range(STEPS):
                            flag, frame = cap.read()
                            if not flag:
                                continue
                            rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            resulthand = signal.process(rgbframe)
                            resultpose = pose.process(cv2.cvtColor(rgbframe, cv2.COLOR_BGR2RGB))
                            try:  # Try two hands in the frame
                                hand1_landmarks.append(resulthand.multi_hand_landmarks[0].landmark)
                                draw.draw_landmarks(image, resulthand.multi_hand_landmarks[0], hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                                draw.draw_landmarks(frame, resulthand.multi_hand_landmarks[0], hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                                hand2_landmarks.append(resulthand.multi_hand_landmarks[1].landmark)
                                draw.draw_landmarks(image, resulthand.multi_hand_landmarks[1], hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                                draw.draw_landmarks(frame, resulthand.multi_hand_landmarks[1], hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                                pose_landmarks.append(resultpose.pose_landmarks.landmark)
                                draw.draw_landmarks(image, resultpose.pose_landmarks, mpose.POSE_CONNECTIONS, POSE_LANDMARKS_STYLE)
                                draw.draw_landmarks(frame, resultpose.pose_landmarks, mpose.POSE_CONNECTIONS, POSE_LANDMARKS_STYLE)
                            except:
                                try:  # Try one hand in the frame
                                    hand1_landmarks.append(resulthand.multi_hand_landmarks[0].landmark)
                                    draw.draw_landmarks(image, resulthand.multi_hand_landmarks[0], hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                                    draw.draw_landmarks(frame, resulthand.multi_hand_landmarks[0], hands.HAND_CONNECTIONS, HAND_LANDMARKS_STYLE, HAND_CONNECTIONS_STYLE)
                                    pose_landmarks.append(resultpose.pose_landmarks.landmark)
                                    draw.draw_landmarks(image, resultpose.pose_landmarks, mpose.POSE_CONNECTIONS, POSE_LANDMARKS_STYLE)
                                    draw.draw_landmarks(frame, resultpose.pose_landmarks, mpose.POSE_CONNECTIONS, POSE_LANDMARKS_STYLE)
                                except:
                                    print(f'Error in Framming - PRINT: {i}')
                                    errors += 1
                                    break
                            images.append(image)
                            live[:, 0:frame.shape[1]] = frame
                            live[:, frame.shape[1]:] = image
                            cv2.imshow('Live', live)

                        # ps, h1 and h2 are the mean of the last STEPS frames
                        ps, h1, h2 = np.zeros(LM_SHAPE[1:]), np.zeros(LM_SHAPE[1:]), np.zeros(LM_SHAPE[1:])
                        for plm, h1lm in zip(pose_landmarks, hand1_landmarks):
                            for i in range(21):
                                ps[i][:] += [plm[i].x, plm[i].y, plm[i].z]
                                h1[i][:] += [h1lm[i].x, h1lm[i].y, h1lm[i].z]
                        ps /= STEPS
                        h1 /= STEPS

                        if len(hand1_landmarks):
                            if len(hand2_landmarks):
                                if np.mean(h1[:, 0]) <= 0.5:
                                    lefthand = h1
                                    righthand = h2
                                else:
                                    lefthand = h2
                                    righthand = h1
                            else:
                                for h2lm in hand2_landmarks:
                                    for i in range(21):
                                        h2[i][:] += [h2lm[i].x, h2lm[i].y, h2lm[i].z]
                                h2 /= STEPS
                                if np.mean(h1[:, 0]) < np.mean(h2[:, 0]):
                                    lefthand = h1
                                    righthand = h2
                                else:
                                    lefthand = h2
                                    righthand = h1

                            data_landmarks = np.array([lefthand, righthand, ps])  # Landmarks Coordinates Data
                            data_image = np.mean([img_to_array(img) for img in images], axis=0)  # Image of Landmarks Data

                            file_name = label + '-' + str(count)
                            cv2.imwrite(os.path.join(dir + '/images/' + label, file_name + '.jpg'), data_image)
                            with open(os.path.join(dir + '/landmarks/' + label, file_name + '.pkl'), 'wb') as file:
                                pickle.dump(data_landmarks, file)
                            count += 1
                            cv2.waitKey(20)
                print(f'Label: {label} have {len(os.listdir(dir + "/images/" + label))} images! - Errors: {errors}')
    cap.release()
    cv2.destroyAllWindows()

# Collect for Train and Validation
collect_data('./data/train', LABELS)
collect_data('./data/validation', LABELS)
