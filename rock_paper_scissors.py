#!/usr/bin/env python3
import rospy
import numpy as np
import random
from sensor_msgs.msg import JointState

import cv2
import mediapipe as mp
from hand_gesture.scripts.preprocess_data import normalize_landmarks, extract_hand_landmarks
import threading

class LeapHandController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('leap_hand_controller', anonymous=False)
        
        # Publisher to the Leap Hand topic
        self.pub = rospy.Publisher('/leaphand_node/cmd_leap', JointState, queue_size=10)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Define the joint positions for rock, paper, and scissors
        self.states = {
            "rock": np.array([3.1416, 4.1888, 4.5553, 4.4157, 3.1416, 4.1190, 5.1487, 4.2412, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157]),
            "paper": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]),
            "scissors": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157])
        }

        # Flag to control the camera feed
        self.running = True

    def publish_state(self, state_name):
        # Create JointState message
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = self.states[state_name].tolist()

        # Publish the message
        self.pub.publish(msg)
        rospy.loginfo(f"Published {state_name} position\n")

    def process_hand_landmarks(self, landmarks):
        # Normalize and extract landmarks using a custom function from hand_gesture
        normalized_landmarks = normalize_landmarks(landmarks)
        hand_data = extract_hand_landmarks(normalized_landmarks)
        rospy.loginfo(f"Extracted hand landmarks: {hand_data}")

    def capture_hand_landmarks(self):
        cap = cv2.VideoCapture(0)  # Open the default camera

        while self.running and cap.isOpened():
            success, image = cap.read()
            if not success:
                rospy.loginfo("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a mirror effect, convert BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and detect hand landmarks
            results = self.hands.process(image)

            # Draw hand landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the image
                    self.mp_draw.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Extract and process hand landmarks
                    self.process_hand_landmarks(hand_landmarks)

            # Show the image with landmarks
            cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        # Start the camera feed in a separate thread
        threading.Thread(target=self.capture_hand_landmarks, daemon=True).start()

        while not rospy.is_shutdown():
            input("Press Enter to change the position (Rock, Paper, or Scissors)...")
            # Choose a random state
            state = random.choice(list(self.states.keys()))
            self.publish_state(state)

        self.running = False  # Stop the camera

if __name__ == "__main__":
    controller = LeapHandController()
    try:
        controller.start()
    except rospy.ROSInterruptException:
        pass
