#!/usr/bin/env python3
import rospy
import numpy as np
import random
from sensor_msgs.msg import JointState

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

    def publish_state(self, state_name):
        # Create JointState message
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = self.states[state_name].tolist()

        # Publish the message
        self.pub.publish(msg)
        rospy.loginfo(f"Published {state_name} position\n")

    def start(self):
        while not rospy.is_shutdown():
            input("Press Enter to change the position (Rock, Paper, or Scissors)...")
            # Choose a random state
            state = random.choice(list(self.states.keys()))
            self.publish_state(state)

if __name__ == "__main__":
    controller = LeapHandController()
    try:
        controller.start()
    except rospy.ROSInterruptException:
        pass
