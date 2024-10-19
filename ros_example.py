#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import rospy

from sensor_msgs.msg import JointState

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
from leap_hand.srv import leap_position, leap_velocity, leap_effort

# This is example code, it reads the position from LEAP Hand and commands it
# Be sure to query the services only when you need it
# This way you don't clog up the communication lines and you get the newest data possible
class Telekinesis:
    def __init__(self):        
        rospy.wait_for_service('/leap_position')
        self.leap_position = rospy.ServiceProxy('/leap_position', leap_position)
        #self.leap_velocity = rospy.ServiceProxy('/leap_velocity', leap_velocity)
        #self.leap_effort = rospy.ServiceProxy('/leap_effort', leap_effort)
        self.pub_hand = rospy.Publisher("/leaphand_node/cmd_ones", JointState, queue_size = 3) 
        r = rospy.Rate(30)   #30hz rate
        while not rospy.is_shutdown():
            r.sleep()
            ###only run these services if you need them.
            curr_pos = np.array(self.leap_position().position)
            #curr_vel = np.array(self.leap_velocity().velocity)
            #curr_eff = np.array(self.leap_effort().effort)
            print(curr_pos)
            ###This is a fresh position, now do some policy stuff here etc.
            
            #Set the position of the hand when you're done
            stater = JointState()
            stater.position = np.array([3.2106218338012695, 4.647961616516113, 4.878058910369873, 3.2719810009002686, 3.069495439529419, 4.477689743041992, 4.878058910369873, 3.4039034843444824, 3.1185829639434814, 4.700117111206055, 4.9271464347839355, 3.2781169414520264, 3.4990100860595703, 1.2655341625213623, 3.6017868518829346, 5.0529327392578125])
            self.pub_hand.publish(stater)  ##choose the right embodiment here
if __name__ == "__main__":
    rospy.init_node("ros_example")
    telekinesis_node = Telekinesis()
