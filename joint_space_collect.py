#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import csv
import os

class LeapHandReader:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('leap_hand_reader', anonymous=False)
        # Subscribe to the joint states topic
        self.joint_state_subscriber = rospy.Subscriber("/leaphand_node/cmd_leap", JointState, self.joint_state_callback)

        # Variable to store the latest joint position
        self.current = None
        self.csv_file = "joint_state.csv"

    def joint_state_callback(self, msg):
        # Extract joint positions from the message
        self.current = msg.position
        # Print the joint positions to the console
        rospy.loginfo(f"Current Joint Positions: {self.current}")

    def save_to_csv(self):
        if self.current is not None:
            # Check if the CSV file exists
            file_exists = os.path.isfile(self.csv_file)
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Joint ' + str(i) for i in range(len(self.current))])
                writer.writerow(self.current)
            rospy.loginfo(f"Saved current joint state to {self.csv_file}")
        else:
            rospy.loginfo("No joint state available to save.")

    def run(self):
        # Keep the node running and wait for user input to save data
        while not rospy.is_shutdown():
            input("Press Enter to save the current joint state to CSV...")
            self.save_to_csv()

if __name__ == "__main__":
    leap_hand_reader = LeapHandReader()
    leap_hand_reader.run()
