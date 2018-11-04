#!/usr/bin/env python

# Rospy dependencies
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

# See a constant tl state for at least three frames before taking action
STATE_COUNT_THRESHOLD = 3

# time in seconds before has_img is switched to 0
IMG_TIMEOUT = 2.0 

DEBUG_ON = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoint_tree = None
        self.has_image = False
        self.prev_img_time = 0.0
        self.tlmap = ('Red','Yellow','Green')
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # Create subscribers
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # Get Traffic light config
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Publisher for upcoming red lights
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        if DEBUG_ON:
            rospy.logwarn('TrafficLight Detector Node initialized')

        self.loop()

    def loop(self):
        # Process traffic light detection with 10 Hz
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            light_wp, state = self.process_traffic_lights()
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1
            rate.sleep()

    def pose_cb(self, msg):
        # /current_pose callback - store pose
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # /base_waypoints callback - store and create KDTree
        self.waypoints = waypoints
        if self.waypoint_tree is None:
            waypoints = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(waypoints)

    def traffic_cb(self, msg):
        # /vehicle/traffic_lights callback
        self.lights = msg.lights

    def image_cb(self, msg):
        # /image_color callback store image and switch state to img recieved
        if not self.has_image:
            rospy.logwarn('Recieved first image, switching traffic light recognition to image analysis')
        self.has_image = True
        self.prev_img_time = rospy.get_time()
        self.camera_image = msg
        
    def get_closest_waypoint(self, x, y):
        
        return self.waypoint_tree.query([x,y],1)[1]

    def get_light_state(self, light):  
        # get state of specific traffic light
        if self.has_image:
            ### No classifier implemented ###
            # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            ## Get classification from image
            #return self.light_classifier.get_classification(cv_image)
            return False
        else:
            return light.state

    def process_traffic_lights(self):
        # return closest traffic light and distance to its stopline
        closest_tl = None
        line_wp_idx = None

        if ((rospy.get_time()-self.prev_img_time) > IMG_TIMEOUT) and self.has_image:
            self.has_image = False
            rospy.logwarn("No further images recieved from camera, using traffic light state provided by Simulation")

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose and self.waypoint_tree and self.waypoints):
            #Get closest waypoint
            car_wp_index = self.get_closest_waypoint(x=self.pose.pose.position.x,y=self.pose.pose.position.y)
            n_waypoints = len(self.waypoints.waypoints)
            
            #Look for next light
            for i, light in enumerate(self.lights):
                tl_wp_index = self.get_closest_waypoint(stop_line_positions[i][0], stop_line_positions[i][1])
                n_wp_to_stopline = tl_wp_index - car_wp_index
                if n_wp_to_stopline >=0 and n_wp_to_stopline < n_waypoints:
                    n_waypoints = n_wp_to_stopline
                    closest_tl = light
                    line_wp_idx = tl_wp_index
        if closest_tl:
            state = self.get_light_state(light)
            if DEBUG_ON:
                rospy.logwarn('Next Traffic Light in: ' + str(n_waypoints) + ' waypoints with state: ' + self.tlmap[state])
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
