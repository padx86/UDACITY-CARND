# SystemIntegration
UDACITY Term 3 Project 3 - Submission

#### Remarks
* Scipy Spatial KDTree is used in the project, so scipy is required
* A classifier or nn for traffic light classification was not implemented due to machine performance.

## Function description
#### 1. waypoint updater
The waypoint_updater node is responsible for the future trajectory planning. It retrieves the entire collection of waypoints provided from the /base_waypoints topic and the current vehicle pose from the /current_pose topic as well as the upcoming traffic light stop line waypoint index from the /traffic_waypoint topic.

From the gathered information it finds the waypoint closest to the car and in the vehicles heading direction and generates the trajectory containing pose, position and velocity information for the next *80* waypoints.

#### 2. twist controller
The twist controllers’ dbw_node generates the vehicles motion controller used to keep the car on the generated trajectory with given velocities and publishes steering, throttle and brake commands. 
The controller consists of a PID controller taking effect on the vehicles longitudinal velocity as well as a yaw controller controlling the vehicles yaw by returning the steering angle. 
The PID Controllers output is then used to set brake (<0) or throttle (>0) commands.

#### 3. traffic light detector
The tl_detector node is responsible for returning the traffic light state as well as the waypoint index of the stop line for the corresponding traffic light. For this purpose it has a subscriber for the /vehicle/traffic_lights topic sending a list of all traffic lights with corresponding stop line and state. Using a KDTree it the finds the traffic light closest to the next traffic light and publishes it to the /traffic_waypoint topic used by the waypoint updater node.

#### 4. traffic light classifier
A traffic light classifier was not implemented as switching on images in the simulation provoked massive performance problems.


## Conclusion
The performance could be enhanced by eliminating jobs done in more than one node like finding the closest waypoint. Instead of processing the closest waypoint in tl_detecor node it could be published in a separate topic.
Additionally yellow traffic lights must be considered in non simulation enviroment.
