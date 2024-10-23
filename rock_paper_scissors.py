#!/usr/bin/env python3
import rospy
import numpy as np
import random
from sensor_msgs.msg import JointState

import os
import cv2
import torch
import pickle
import mediapipe as mp
from hand_gesture.scripts.model import HGRModel

import time
import threading
from collections import Counter

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Construct the paths relative to the script's location
model_path = os.path.join(base_dir, 'hand_gesture/models/hgr_model.pth')
label_encodings_path = os.path.join(base_dir, 'hand_gesture/models/label_encodings.pkl')

class LeapHandController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('leap_hand_controller', anonymous=False)

        # Publisher to the Leap Hand topic
        self.pub = rospy.Publisher('/leaphand_node/cmd_leap', JointState, queue_size=10)

        # Define the joint positions for rock, paper, and scissors
        self.states = {
            "rock": np.array([3.1416, 4.1888, 4.5553, 4.4157, 3.1416, 4.1190, 5.1487, 4.2412, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157]),
            "paper": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]),
            "scissors": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157])
        }

        # Score tracking
        self.user_score = 0
        self.computer_score = 0

        # Load the trained HGR model and label encodings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HGRModel(in_features=21*2, out_features=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        with open(label_encodings_path, 'rb') as f:
            self.label_to_index = pickle.load(f)
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        # Threading
        self.frame = None
        self.thread_running = True
        self.lock = threading.Lock()

    def publish_state(self, state_name):
        # Create JointState message
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = self.states[state_name].tolist()

        # Publish the message
        self.pub.publish(msg)
        rospy.loginfo(f"Published {state_name} position")

    def extract_landmarks(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            return hand_landmarks.landmark
        return None

    def normalize_landmarks(self, landmarks, image_shape):
        normalized_landmarks = []
        image_height, image_width, _ = image_shape
        for landmark in landmarks:
            normalized_landmarks.append([
                min(int(landmark.x * image_width), image_width - 1),
                min(int(landmark.y * image_height), image_height - 1)
            ])
        normalized_landmarks = np.array(normalized_landmarks, dtype=np.float32)
        normalized_landmarks = normalized_landmarks - normalized_landmarks[0]
        normalized_landmarks = normalized_landmarks / np.max(np.abs(normalized_landmarks))
        return normalized_landmarks.flatten()

    def preprocess_frame(self, frame):
        landmarks = self.extract_landmarks(frame)
        if landmarks is None:
            return None
        normalized_landmarks = self.normalize_landmarks(landmarks, frame.shape)
        return torch.tensor(normalized_landmarks, dtype=torch.float32).to(self.device).unsqueeze(0)
    
    def camera_thread(self):
        cap = cv2.VideoCapture(0)
        self.countdown_flag = False
        self.countdown_start_time = None

        while self.thread_running:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

                if self.countdown_flag:
                    elapsed_time = time.time() - self.countdown_start_time
                    countdown_number = 3 - int(elapsed_time)
                    if countdown_number > 0:
                        cv2.putText(frame, str(countdown_number), (frame.shape[1] // 2, frame.shape[0] // 2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10, cv2.LINE_AA)
                    else:
                        self.countdown_flag = False

                cv2.imshow("Webcam Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.thread_running = False

        cap.release()
        cv2.destroyAllWindows()

    def start_countdown(self):
        self.countdown_flag = True
        self.countdown_start_time = time.time()

    def capture_and_predict(self, duration=0.5):
        predictions = []
        capture_start_time = time.time()

        while time.time() - capture_start_time < duration:
            frame = self.get_frame()
            if frame is None:
                continue

            # Process the frame
            input_tensor = self.preprocess_frame(frame)
            if input_tensor is None:
                continue

            # Make predictions
            with torch.no_grad():
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                predictions.append(self.index_to_label[predicted.item()].lower())

        return predictions

    def get_frame(self):
        with self.lock:
            return self.frame

    def determine_winner(self, user_move, computer_move):
        rules = {
            'rock': 'scissors',
            'scissors': 'paper',
            'paper': 'rock'
        }
        if user_move == computer_move:
            return 'Draw'
        elif rules[user_move] == computer_move:
            return 'User'
        else:
            return 'Computer'
        
    def play_competitive(self):
        while True:
            # Press Enter to play
            input("Press Enter to play a round of Rock, Paper, Scissors or type 'quit' to exit: ").strip().lower()
            if input == 'quit':
                break
            
            # Countdown
            self.start_countdown()
            time.sleep(3)

            # Capture multiple frames and predict hand gestures
            predictions = self.capture_and_predict(duration=0.5)

            if not predictions:
                print("No hand detected. Try again.")
                continue

            # Get the mode of the predictions
            user_move = Counter(predictions).most_common(1)[0][0]
            print(f"User Move: {user_move}")

            # Determine the winning move for Leap Hand
            if user_move == 'rock':
                leap_hand_move = 'paper'
            elif user_move == 'paper':
                leap_hand_move = 'scissors'
            else:
                leap_hand_move = 'rock'

            self.publish_state(leap_hand_move)
            print(f"Leap Hand Move: {leap_hand_move}")

            # Determine the winner
            winner = self.determine_winner(user_move, leap_hand_move)

            # Update and display scores
            if winner == 'User':
                self.user_score += 1
            elif winner == 'Computer':
                self.computer_score += 1

            print(f"Winner: {winner}")
            print(f"Score -> User: {self.user_score} | Computer: {self.computer_score}\n")

    def play_casual(self):
        while True:
            # Press Enter to play
            input("Press Enter to play a round of Rock, Paper, Scissors or type 'quit' to exit: ").strip().lower()
            if input == 'quit':
                break
            
            # Countdown
            self.start_countdown()
            time.sleep(3)

            # Computer (Leap Hand) chooses a random state
            leap_hand_move = random.choice(list(self.states.keys()))
            self.publish_state(leap_hand_move)
            print(f"Leap Hand Move: {leap_hand_move}")

            # Capture multiple frames and predict hand gestures
            predictions = self.capture_and_predict(duration=0.5)

            if not predictions:
                print("No hand detected. Try again.")
                continue

            # Get the mode of the predictions
            user_move = Counter(predictions).most_common(1)[0][0]
            print(f"User Move: {user_move}")

            # Determine the winner
            winner = self.determine_winner(user_move, leap_hand_move)

            # Update and display scores
            if winner == 'User':
                self.user_score += 1
            elif winner == 'Computer':
                self.computer_score += 1

            print(f"Winner: {winner}")
            print(f"Score -> User: {self.user_score} | Computer: {self.computer_score}\n")

    def start(self):
        # Start the camera thread
        camera_thread = threading.Thread(target=self.camera_thread)
        camera_thread.start()

        while not rospy.is_shutdown():
            # Ask the player for competitive or casual play
            mode = input("Choose mode: 'competitive' or 'casual': ").strip().lower()

            if mode == 'competitive':
                self.play_competitive()
            elif mode == 'casual':
                self.play_casual()
            else:
                print("Invalid mode. Please choose 'competitive' or 'casual'.")

            # Display the final scores
            print(f"Final Score -> User: {self.user_score} | Computer: {self.computer_score}")

            # Reset the scores
            self.user_score = 0
            self.computer_score = 0

        # Stop the camera thread
        self.thread_running = False
        camera_thread.join()

if __name__ == "__main__":
    controller = LeapHandController()
    try:
        controller.start()
    except rospy.ROSInterruptException:
        pass
