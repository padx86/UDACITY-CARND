#ifndef PATHPLANNER_H
#define PATHPLANNER_H

#include <vector>
#include <iostream>
#include <math.h>
#include "helpers.h"
#include "spline.h"
#include "trafficobject.h"

using std::vector;
using std::pair;

class pathplanner {
private:
	
	enum BEHAVIOUR {
		LC_LEFT,
		LC_RIGHT,
		FOLLOW
	};

	int counter;

	//Generel vehicle data
	const double max_acc_ = 2.0;					// Maximum possible accelaration in meters/second
	const double min_acc_ = -6.0;					// Minimum possible barking in meters/second
	const double max_vel_ = mph2mps(49.5);			// Maximum velocity to not exceed speed limit
	const double min_vel_ = mph2mps(0.0);			// Minimum velocity (no reversing on highways
	const double max_steer_angle_ = deg2rad(20);	// Save steer angle for driving on highways
	const double min_steer_angle_ = deg2rad(-20);	// Save steer angle for driving on highways
	
	//Generel enviroment data
	double lane_width_;				// width of lane
	int prev_path_size_;			// previous path size
	float delta_t_;					// delta_t = .02 for this simulator
	double ref_vel_;				// reference velocity
	double lane_index_;				// current lane index
	double desired_lane_;			// desired lane

	//Vehcile states
	double car_x_;					// x-coordinate of car
	double car_y_;					// y-coordinate of car
	double car_yaw_;				// global yaw angle of car
	double car_vel_;				// Total velocity of car
	double car_s_;					// Frenet longitudinal distance
	double car_d_;					// frenet lateral distance
	bool forward_vehicle_detected;	// True if vehicle within specific range on same lane, NOT IN USE IN SUBMISSION
	//Previous values
	vector<double> previous_path_x_;	// x-values of  previous path
	vector<double> previous_path_y_;	// y values of previous path
	double end_path_s_;					// last value of frenet longitudinal distance
	double end_path_d_;					// last value of frenet lateral distance

	//Other vehicles
	vector<vector<double>> sensor_fusion_;	// Sensor fusion objects
	vector<trafficobject> traffic;			// List of trafficobjects
	trafficobject to_front;					// Traffic Object in front of ego-vehicle on same lane
	trafficobject to_front_left;			// Traffic Object in front of ego-vehicle on left lane relative to ego-vehicle
	trafficobject to_front_right;			// Traffic Object in front of ego-vehicle on right lane relative to ego-vehicle
	trafficobject to_back_left;				// Traffic Object in back of ego-vehicle on left lane relative to ego-vehicle
	trafficobject to_back_right;			// Traffic Object in back of ego-vehicle on right lane relative to ego-vehicle

	//Cost function variables
	unsigned int lc_punish_counter;			// Counter to add cost directly after lane change
	const unsigned int lc_punish_max = 4;	// Value lc_punish_counter is set to on lane change
	
	vector<double> map_waypoints_x_;		// x-coordinates of waypoints
	vector<double> map_waypoints_y_;		// y coordinates of waypoints
	vector<double> map_waypoints_s_;		// s-coordinates of waypoints
	vector<double> map_waypoints_dx_;		// relative x-coordinates of waypoints
	vector<double> map_waypoints_dy_;		// relative y-coordinates of waypoints

	void updateTrafficObjects();			// Updates list of traffic objects
	BEHAVIOUR decide();						// Calculates costs and decides what action to take

public:
	
	pathplanner();																		// Constructor
	void addWaypoint(double x, double y, double s, double dx, double dy);				// Adds a waypoint to the the private map_waypoint_ vectors
	void updateMeassurement(double car_x, double car_y, double car_yaw, double car_vel, 
		double car_s, double car_d, vector<double> previous_path_x, 
		vector<double> previous_path_y, double end_path_s, double end_path_d, 
		vector<vector<double>> sensor_fusion);											// Updates a given measurement from simulator
	pair<vector<double>, vector<double>> drive(double predictrion_horizon);				// Returns trajectory for optimal driving
	pair<vector<double>, vector<double>> driveStraight(int n_points, double dist);		// Returns trjacetory for driving straight
	pair<vector<double>, vector<double>> driveFrenet(int n_points, double dist);		// Returns trajectory for driving on ego lane
	pair<vector<double>, vector<double>> driveSpline(vector<double> waypoint_dist);		// Returns smooth trajectory
	void getRelevantTraffic();															// Updates relevant traffic objects
	void predictTraffic();																// Predicts all monitored traffic objects
	void checkForwardVehicle();															// Checks if there is a vehicle in front of the car
	void speedControl();																// Controls vehicles speed depending on forward object;
	void debug();																		// Function to cout class properties
};

#endif // !PATHPLANNER_H

