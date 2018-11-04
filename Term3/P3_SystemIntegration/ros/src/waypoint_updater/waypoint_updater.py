#!/usr/bin/env python

#ROS dependencies
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32

# External dependencies
from scipy.spatial import KDTree
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 80 # Number of waypoints we will publish. You can change this number
DEBUG_ON = True
REACT_TO_TRAFFIC_LIGHTS = True

class WaypointUpdater(object):
    def __init__(self):
        
        ## Initialize node
        rospy.init_node('waypoint_updater')

        ## Initialize Variables
        
        # Waypoints
        self.base_waypoints = None
        self.waypoint_tree = None
        self.previous_closest_waypoint_index = -1         
        self.previous_velocitys = []
        self.STATE_DECELARATING = False
        self.decelaration = 0.0
        # Trafficlights
        self.stopline_wp_index = None

        # Vehicle states
        self.car_x = None
        self.car_y = None
        self.car_orientation = None
        self.current_velocity = 0.0

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        
        # Publishers
        self.wp_publisher = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        if DEBUG_ON:
            rospy.logwarn('Waypointupdater is initialized, looking ahead ' + str(LOOKAHEAD_WPS) + ' waypoints')

        self.loop()

    ## Main loop function
    def loop(self):
        ## Use loop function to give control on publishing frequency
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.car_x and self.base_waypoints:
                cl_wp_idx = self.get_closest_waypoint_index()
                self.publish_waypoints(cl_wp_idx)
            rate.sleep()

    def publish_waypoints(self, closest_waypoint_index):
        # publishes next waypoints with velocity values
        if REACT_TO_TRAFFIC_LIGHTS:
            msg = self.generate_tl_wp(closest_waypoint_index)
        else:
            msg = Lane()
            msg.header = self.base_waypoints.header
            msg.waypoints = self.base_waypoints.waypoints[closest_waypoint_index:closest_waypoint_index + LOOKAHEAD_WPS]

        self.wp_publisher.publish(msg)
        
    def generate_tl_wp(self,closest_waypoint_index):
        # Returns msg of type Lane if traffic lights should be taken into consideration

        # Create Message
        msg = Lane()
        msg.header = self.base_waypoints.header

        # Get next LOOKAHEAD_WPS
        last_waypoint_index = closest_waypoint_index + LOOKAHEAD_WPS
        waypoints = self.base_waypoints.waypoints[closest_waypoint_index:last_waypoint_index]

        if self.stopline_wp_index == -1 or last_waypoint_index <= self.stopline_wp_index:
            # Stopline to far, behind vehicle or traffic light state not red
            msg.waypoints = waypoints
            self.STATE_DECELARATING = False
        else:
            # Index of stop line relative to closest waypoint
            stop_index = self.stopline_wp_index - closest_waypoint_index 
            
            # Calculate linear decelaration for first entering stop light detection progress
            if self.STATE_DECELARATING == False:
                initial_velocity = waypoints[0].twist.twist.linear.x
                if stop_index > 10:
                    # Deceleration per waypoint if stop line more than 10 Waypoints ahead at first detection as no "Yellow-Detection" implemented
                    self.decelaration = initial_velocity/(stop_index-1) 
                else:
                    # Do not decelarate if red light appears during car is to close
                    self.decelaration = 0
            else:
                # Get decelarating initial velocity at closest waypoint
                initial_velocity = self.previous_velocitys[closest_waypoint_index-self.previous_closest_waypoint_index]
            
            # Clear previously stored velocities and create waypoints list
            decleration_waypoints = []
            self.previous_velocitys = []
            
            # Generate waypoitns
            for i, wp in enumerate(waypoints):
                p = Waypoint()
                p.pose = wp.pose
                decelaration_velocity = initial_velocity - self.decelaration*i
                if decelaration_velocity < 1.0:
                    decelaration_velocity = 0.0
                p.twist.twist.linear.x = decelaration_velocity
                self.previous_velocitys.append(decelaration_velocity)
                decleration_waypoints.append(p)
            msg.waypoints = decleration_waypoints
            self.STATE_DECELARATING = True
        self.previous_closest_waypoint_index = closest_waypoint_index

        return msg

    ## Callbacks
    def pose_cb(self, msg):
        ## Callback for /current_pose topic subscription
        self.car_x = msg.pose.position.x
        self.car_y = msg.pose.position.y
        self.car_yaw = self.get_yaw_from_quaternion(msg.pose.orientation)
    
    def waypoints_cb(self, waypoints):
        ## Callback for /base_waypoints topic subscription
        self.base_waypoints = waypoints
        if self.waypoint_tree is None:
            waypoints = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(waypoints)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint subscription
        self.stopline_wp_index = msg.data

    ## Helper functions
    def get_closest_waypoint_index(self):
        # returns closest waypoint from KDTree
        if self.waypoint_tree is not None:
            closest_idx = self.waypoint_tree.query([self.car_x, self.car_y],1)[1]
            wp_angle = math.atan2((self.base_waypoints.waypoints[closest_idx].pose.pose.position.y-self.car_y),(self.base_waypoints.waypoints[closest_idx].pose.pose.position.x-self.car_x))
            diff_angle = abs(self.car_yaw - wp_angle)
            if (diff_angle > (math.pi / 4)):
                closest_idx = closest_idx + 1
        else:
            rospy.logwarn('KDTree has not been initialized yet')
            closest_idx = 0

        return closest_idx

    def distance(self, waypoints, wp1, wp2):
        # returns piecewise distance between 2 waypoints
        if len(waypoints)-1 < wp1 or len(waypoints)-1 < wp2:
            rospy.logwarn('Distance function indices out of range')
            return -1
            
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def get_yaw_from_quaternion(self, quat_orientation):
        # returns yaw angle in radians from quaternion
        return math.atan2(2.0*(quat_orientation.x+quat_orientation.y+quat_orientation.w*quat_orientation.z),1.0-2.0*(quat_orientation.y**2+quat_orientation.z**2))

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
