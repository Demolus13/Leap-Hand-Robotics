#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from leap_hand.srv import leap_position
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu

class LeapHandController:
    def _init_(self):
        # Initialize node
        rospy.init_node("leap_hand_controller")
        
        # Wait for LEAP hand position service to be available
        rospy.wait_for_service('/leap_position')
        self.leap_position_service = rospy.ServiceProxy('/leap_position', leap_position)

        # Publisher to command the LEAP hand
        self.pub_hand = rospy.Publisher("/leaphand_node/cmd_leap", JointState, queue_size=3)

        # Connect to the dynamixel motor client
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.dxl_client = self.connect_to_motors(motors)
        
        rospy.loginfo("LeapHandController Initialized.")

    def connect_to_motors(self, motors):
        # Try to connect to the motor
        try:
            dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            dxl_client.connect()
        except Exception:
            rospy.logwarn("Could not connect to motors on USB0, trying USB1")
            try:
                dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                dxl_client.connect()
            except Exception:
                rospy.logwarn("Could not connect to motors on USB1, trying USB2")
                dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', 4000000)
                dxl_client.connect()
        return dxl_client

    def get_desired_position(self):
        try:
            # Query the current LEAP hand position from the service
            curr_pos = self.leap_position_service().position
            return np.array(curr_pos)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return None

    def move_hand_to_position(self, position):
        # Create a JointState message to send the desired position
        joint_state = JointState()
        joint_state.position = position
        
        # Publish the desired position to the hand controller
        self.pub_hand.publish(joint_state)
        rospy.loginfo(f"Moving hand to position: {position}")
        
        # Send command to motors directly
        self.dxl_client.write_desired_pos(self.motors, position)

    def run(self):
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            # Fetch the desired position (could come from a different source)
            desired_position = self.get_desired_position()
            if desired_position is not None:
                # Command the LEAP hand to move to the desired position
                self.move_hand_to_position(desired_position)
            
            rate.sleep()

if __name__ == "__main__":
    controller = LeapHandController()
    controller.run()