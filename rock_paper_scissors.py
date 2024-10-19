#!/usr/bin/env python3
import numpy as np
import rospy
import random

from sensor_msgs.msg import JointState

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from leap_hand.srv import *

class RockPaperScissors:
    def __init__(self):        
        rospy.wait_for_service('/leap_position')
        self.leap_position = rospy.ServiceProxy('/leap_position', leap_position)
        self.pub_hand = rospy.Publisher("/leaphand_node/cmd_ones", JointState, queue_size=3) 
        self.states = {
            "rock": np.array([3.1416, 4.1888, 4.5553, 4.4157, 3.1416, 4.1190, 5.1487, 4.2412, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157]),
            "paper": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416]),
            "scissors": np.array([3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 4.2237, 4.7124, 4.4506, 2.6005, 1.5184, 4.6775, 4.4157])
        }

        r = rospy.Rate(30)  # 30hz rate
        while not rospy.is_shutdown():
            r.sleep()
            input("Press Enter to change state (Rock, Paper, Scissors): ")
            self.set_random_state()

    def set_random_state(self):
        state_name = random.choice(list(self.states.keys()))
        state = self.states[state_name]
        print(f"Setting state: {state_name}")

        # Publish the joint state to the hand
        stater = JointState()
        stater.position = state
        self.pub_hand.publish(stater)
        print(f"Published state: {state_name}")

if __name__ == "__main__":
    rospy.init_node("rock_paper_scissors_node")
    rps_node = RockPaperScissors()
