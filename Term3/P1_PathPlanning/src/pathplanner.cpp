#include "pathplanner.h"

void pathplanner::updateTrafficObjects()
{
	/* UPDATE TRAFFIC OBJECTS IN traffic PROPERTY */
	for (vector<vector<double>>::iterator fo = this->sensor_fusion_.begin(); fo != this->sensor_fusion_.end(); fo++) {
		bool fo_found = false;
		for (vector<trafficobject>::iterator to = this->traffic.begin(); to != this->traffic.end(); to++) {
			if (fo->at(0) == to->getId()) {
				fo_found = true;
				to->update(fo->at(5), fo->at(6), sqrt(pow(fo->at(3), 2) + pow(fo->at(4), 2)), this->car_s_, this->car_d_, this->car_vel_,50-this->prev_path_size_);
				break;
			}
		}
		if (!fo_found) {
			trafficobject tr_obj = trafficobject(fo->at(0), this->lane_width_, this->delta_t_);
			tr_obj.update(fo->at(5), fo->at(6), sqrt(pow(fo->at(3), 2) + pow(fo->at(4), 2)), this->car_s_, this->car_d_, this->car_vel_, 50-this->prev_path_size_);
			this->traffic.push_back(tr_obj);
		}
	}
}

pathplanner::BEHAVIOUR pathplanner::decide()
{
	/* DECIDE WHAT ACTION TO TAKE NEXT */
	BEHAVIOUR what_to_do;

	double follow_cost = 0.;
	double lc_left_cost = 0.;
	double lc_right_cost = 0.;


		
	if (this->desired_lane_ != this->lane_index_) {
		return BEHAVIOUR::FOLLOW;					// If lane change in progress, do not make new decission
	}

	// Calculate cost
	// 1. Cost to stay on lane
	double speed_cost = this->to_front.s_predicted < 2.5 * this->max_vel_ ? (-1)*this->to_front.s_dot_predicted / this->max_vel_ : 0;	// Define cost related to forward vehicle speed
	double dist_cost = (1.5 * this->ref_vel_ - this->to_front.s_predicted) / (this->ref_vel_);											// Define cost related to distance to forward vehicle
	dist_cost = dist_cost > 0. ? dist_cost : 0.;
	follow_cost = .4 * dist_cost + .6 * speed_cost;		//
	follow_cost = follow_cost < 0 ? 0 : follow_cost;	//Cap cost lower 0 
	follow_cost = follow_cost > 1 ? 1 : follow_cost;	//Cap cost higher 1

	double lc_counter_cost = this->lc_punish_counter / (1.0 * this->lc_punish_max); //Punish previous lane changes

	//2. Cost to switch lane
	if (this->lane_index_ == 0) {
		lc_left_cost = 1.;
	}
	else {
		double s_front_left = abs(this->to_front_left.s_predicted) >.001 ? 1 - exp(-30 / abs(this->to_front_left.s_predicted)) : 1.;	// Punish vehicle on left front side
		double s_back_left = abs(this->to_back_left.s_predicted) >.001 ? 1 - exp(-15 / abs(this->to_back_left.s_predicted)) : 1.;		// Punish vehicle on left back side 
		double s_dot_front_left = this->to_front_left.s_dot_predicted < 0 ? -this->to_front_left.s_dot_predicted/this->max_vel_ : 0;	// Punish slow heading vehicle on left lane
		double s_dot_back_left = this->to_back_left.s_dot_predicted > 0 ? this->to_back_left.s_dot_predicted / this->max_vel_ : 0;		// Punish fast rear vehicle on left lane 
		lc_left_cost = .5*s_front_left + .5*s_back_left + .3*s_dot_front_left + .3*s_dot_back_left + .0 * lc_counter_cost;				// Define cost for left lane change
		lc_left_cost = lc_left_cost < 0 ? 0 : lc_left_cost;
		lc_left_cost = lc_left_cost > 1 ? 1 : lc_left_cost;
	}
	if (this->lane_index_ == 2) {
		lc_right_cost = 1.;
	}
	else {
		double s_front_right = abs(this->to_front_right.s_predicted) >.001 ? 1 - exp(-30 / abs(this->to_front_right.s_predicted)) : 1.;			// Punish vehicle on right front side
		double s_back_right = abs(this->to_back_right.s_predicted) >.001 ? 1 - exp(-15 / abs(this->to_back_right.s_predicted)) : 1.;			// Punish vehicle on right back side 
		double s_dot_front_right = this->to_front_right.s_dot_predicted < 0 ? -this->to_front_right.s_dot_predicted/this->max_vel_ : 0;			// Punish slow heading vehicles on right lane, reward fast vehicles
		double s_dot_back_right = this->to_back_right.s_dot_predicted > 0 ? this->to_back_right.s_dot_predicted / this->max_vel_ : 0;			// Punish fast rear vehicle on right lane 
		lc_right_cost = .5*s_front_right + .5*s_back_right + .3*s_dot_front_right + .3*s_dot_back_right + .0 * lc_counter_cost;					// Define cost for right lane change
		lc_right_cost = lc_right_cost < 0 ? 0 : lc_right_cost;																					// Cap cost lower 0 
		lc_right_cost = lc_right_cost > 1 ? 1 : lc_right_cost;																					// Cap cost higher 1
	}

	//Compare cost and decide what to do
	if (lc_left_cost < follow_cost && lc_left_cost < lc_right_cost) {
		what_to_do = BEHAVIOUR::LC_LEFT;
		this->lc_punish_counter = this->lc_punish_max;
		this->desired_lane_ = this->lane_index_ - 1;
	}
	else if (lc_right_cost < follow_cost && lc_right_cost < lc_left_cost) {
		what_to_do = BEHAVIOUR::LC_RIGHT;
		this->lc_punish_counter = this->lc_punish_max;
		this->desired_lane_ = this->lane_index_ + 1;
	}
	else {
		what_to_do = BEHAVIOUR::FOLLOW;
		if (this->lc_punish_counter != 0) {
			this->lc_punish_counter--;
		}
	}
	std::cout << "LCL " << lc_left_cost << "  LK " << follow_cost << "  LCR " << lc_right_cost << std::endl; // Output cost
	return what_to_do;
}

pathplanner::pathplanner()
{
	/* CONSTRUCTOR */
	this->lane_width_ = 4.;									// 4 meters lane width
	this->ref_vel_ = 1.;									// Initial reference velocity
	this->lane_index_ = 1;									// Initial Lane Index
	this->prev_path_size_ = 0;								// Init size of previous path 
	this->delta_t_ = .02;									// Delta t (position update rate of simulator) 
	this->forward_vehicle_detected = false;					// True if relevant forward vehicle is detected
	this->counter = 0;										// Initialize counter
	this->desired_lane_ = -1;								// Initialize desired lane
	this->lc_punish_counter = this->lc_punish_max;			// Initialize punish counter so car does not switch lanes on init
}

void pathplanner::addWaypoint(double x, double y, double s, double dx, double dy)
{
	/* SAVE WAYPOINTS TO CLASS INSTANCE */
	this->map_waypoints_x_.push_back(x);
	this->map_waypoints_y_.push_back(y);
	this->map_waypoints_s_.push_back(s);
	this->map_waypoints_dx_.push_back(dx);
	this->map_waypoints_dy_.push_back(dy);
}

void pathplanner::updateMeassurement(double car_x, double car_y, double car_yaw, double car_vel, double car_s, double car_d, vector<double> previous_path_x, vector<double> previous_path_y, double end_path_s, double end_path_d, vector<vector<double>> sensor_fusion)
{
	/* UPDATE EGO AND FOREIGN VEHICLES */
	this->car_x_ = car_x;
	this->car_y_ = car_y;
	this->car_yaw_ = car_yaw;
	this->car_vel_ = car_vel;
	this->car_s_ = car_s;
	this->car_d_ = car_d;
	this->previous_path_x_ = previous_path_x;
	this->previous_path_y_ = previous_path_y;
	this->end_path_s_ = end_path_s;
	this->end_path_d_ = end_path_d;
	this->lane_index_ = floor(car_d / this->lane_width_);
	if (this->desired_lane_ == -1) {
		this->desired_lane_ = this->lane_index_;
	}
	
	this->sensor_fusion_ = sensor_fusion;

	this->prev_path_size_ = previous_path_x.size();
	this->updateTrafficObjects();
}

pair<vector<double>, vector<double>> pathplanner::drive(double prediction_horizon)
{
	/* MAIN PATHPLANNING FUNCTION */
	pair<vector<double>, vector<double>> coordinates;
	BEHAVIOUR what_to_do;

	/// Run every Loop
	this->getRelevantTraffic();						// Update surrounding vehicles
	this->to_front.predict(prediction_horizon);		// Predict forward vehicle states

	if (this->counter % (int)floor(prediction_horizon/(2*this->delta_t_)) == 0) {
		/// Do not Run every Loop to reduce performance -- 
		// Predict surrounding vehicles, calculate cost and decide what to do
		this->to_front_left.predict(prediction_horizon);
		this->to_front_right.predict(prediction_horizon);
		this->to_back_left.predict(prediction_horizon);
		this->to_back_right.predict(prediction_horizon);
		this->decide();
		this->debug();
	}
	if (this->counter >= 2*(int)floor(prediction_horizon /(2*this->delta_t_))) {
		this->counter = 0;				// Reset counter
	}
	this->counter++;

	// Generate anchor distances for the spline to use depending on vehicle velocity
	vector<double> waypoints;
	waypoints.push_back(2.5*(this->car_vel_> 10 ? this->car_vel_ : 10));
	waypoints.push_back(5* (this->car_vel_ > 10 ? this->car_vel_ : 10));
	waypoints.push_back(7.5* (this->car_vel_ > 10 ? this->car_vel_ : 10));
	coordinates = this->driveSpline(waypoints);
	return coordinates;
}

pair<vector<double>, vector<double>> pathplanner::driveStraight(int n_points, double dist)
{
	/* DRIVE STRAIGHT FUNCTION FROM PROJECT WALKTHROUGH - NOT IN USE */
	pair<vector<double>, vector<double>> coordinates;
	
	for (int i = 0; i < n_points; i++) {
		coordinates.first.push_back(this->car_x_ + (i*dist)*cos(this->car_yaw_));
		coordinates.second.push_back(this->car_y_ + (i*dist)*sin(this->car_yaw_));
	}

	return coordinates;
}

pair<vector<double>, vector<double>> pathplanner::driveFrenet(int n_points, double dist)
{
	/* DRIVE FRENET FUNCTION FROM PROJECT WALKTHROUGH - NOT IN USE */
	pair<vector<double>, vector<double>> coordinates;
	for (int i = 0; i < n_points; i++) {
		double next_s = this->car_s_ + (i + 1)*dist;
		double next_d = (this->lane_index_ + .5) * this->lane_width_;
		vector<double> xy = getXY(next_s,next_d,this->map_waypoints_s_,this->map_waypoints_x_, this->map_waypoints_y_);
		coordinates.first.push_back(xy[0]);
		coordinates.second.push_back(xy[1]);
	}
	return coordinates;
}

pair<vector<double>, vector<double>> pathplanner::driveSpline(vector<double> waypoint_dist)
{
	/* DRIVE SPLINE FUNCTION FROM PROJECT WALKTHROUGH MODIFIED TO WORK ON THIS PROJECT */
	pair<vector<double>, vector<double>> coordinates;
	
	vector<double> ptsx;
	vector<double> ptsy;

	tk::spline s;
	
	double ref_x = this->car_x_;
	double ref_y = this->car_y_;
	double ref_yaw = this->car_yaw_;

	if (this->prev_path_size_ < 2) {
		// Create coordinates if previous coordinates returned by simulator is less than 2
		double prev_car_x = this->car_x_ - cos(this->car_yaw_);	// Create a x-point in the cars heading direction behind the car
		double prev_car_y = this->car_y_ - sin(this->car_yaw_);	// Create a y-point in the cars heading direction behind the car 

		ptsx.push_back(prev_car_x);
		ptsx.push_back(this->car_x_);

		ptsy.push_back(prev_car_y);
		ptsy.push_back(this->car_y_);
	}
	else {
		// Use last 2 coordinates from simulator to create begin of spline 
		ref_x = this->previous_path_x_[this->prev_path_size_ - 1];
		ref_y = this->previous_path_y_[this->prev_path_size_ - 1];
		double ref_x_prev = this->previous_path_x_[this->prev_path_size_ - 2];
		double ref_y_prev = this->previous_path_y_[this->prev_path_size_ - 2];
		ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

		ptsx.push_back(ref_x_prev);
		ptsx.push_back(ref_x);

		ptsy.push_back(ref_y_prev);
		ptsy.push_back(ref_y);
	}
	// Generate anchor points in XY coordinates
	for (vector<double>::iterator it = waypoint_dist.begin(); it != waypoint_dist.end(); it++) {
		vector<double> wp = getXY(this->car_s_ + *it, (this->desired_lane_ + .5) * this->lane_width_, this->map_waypoints_s_, this->map_waypoints_x_, this->map_waypoints_y_);
		ptsx.push_back(wp[0]);
		ptsy.push_back(wp[1]);
	}

	// Transform to cars local coordinates
	pair<vector<double>,vector<double>> local_coordinates = getLocalTransformation(ref_x, ref_y, ref_yaw, ptsx, ptsy); 
	
	// Set spline points
	s.set_points(local_coordinates.first, local_coordinates.second);

	//Build new coordinates vectors, start wirh 
	for (int i = 0; i < this->prev_path_size_; i++) {
		coordinates.first.push_back(this->previous_path_x_[i]);
		coordinates.second.push_back(this->previous_path_y_[i]);
	}

	double target_x = 30.0;
	double target_y = s(target_x);
	double target_dist = distance(target_x, target_y, 0, 0);

	double x_add_on = 0;

	for (int i = 1; i <= 50 - this->prev_path_size_; i++) {
		// Generate smooth coordinates for remaining coordinates
		this->speedControl();
		double N = (target_dist/(this->delta_t_*this->ref_vel_));
		double x_point = x_add_on + target_x / N;
		double y_point = s(x_point);
		
		x_add_on = x_point;

		double x_ref = x_point;
		double y_ref = y_point;

		//Retransform to global
		x_point = (x_ref*cos(ref_yaw) - y_ref * sin(ref_yaw));
		y_point = (x_ref*sin(ref_yaw) + y_ref * cos(ref_yaw));
		x_point += ref_x;
		y_point += ref_y;
		coordinates.first.push_back(x_point);
		coordinates.second.push_back(y_point);

	}

	return coordinates;
}

void pathplanner::getRelevantTraffic()
{
	this->to_back_left.s_ = -1000.;
	this->to_back_right.s_ = -1000.;
	this->to_front.s_ = 1000.;
	this->to_front_left.s_ = 1000.;
	this->to_front_right.s_ = 1000.;

	
	// Extract relevant traffic objects 
	// Use vehicle infront on same lane, 2 vehicles right and 2 vehicles left
	for (vector<trafficobject>::iterator to = this->traffic.begin(); to != this->traffic.end(); to++)
	{
		if ((to->relative_lane_index == 0) && (to->s_ < this->to_front.s_) && (to->s_ > 0)) {
			this->to_front = *to;
		}
		if ((to->relative_lane_index == 1) && (to->s_ < this->to_front_right.s_) && (to->s_ > 0)) {
			this->to_front_right = *to;
		}
		if ((to->relative_lane_index == 1) && (to->s_ > this->to_back_right.s_) && (to->s_ < 0)) {
			this->to_back_right = *to;
		}
		if ((to->relative_lane_index == -1) && (to->s_ < this->to_front_left.s_) && (to->s_ > 0)) {
			this->to_front_left = *to;
		}
		if ((to->relative_lane_index == -1) && (to->s_ > this->to_back_left.s_) && (to->s_ < 0)) {
			this->to_back_left = *to;
		}
	}
}

void pathplanner::predictTraffic()
{
	//Predict all Traffic given by Simulator - NOT IN USE
	for (vector<trafficobject>::iterator to = this->traffic.begin(); to != this->traffic.end(); to++) {
		if (frenetDistance(this->car_s_, to->s_)) {
			to->predict(50);
		}
	}
}

void pathplanner::checkForwardVehicle()
{
	//First attempt speed control - NOT IN USE
	this->forward_vehicle_detected = false;
	if (this->prev_path_size_ > 0) {
		this->car_s_ = this->end_path_s_;
	}

	for (vector<vector<double>>::iterator it = this->sensor_fusion_.begin(); it != this->sensor_fusion_.end(); it++) {
		float d = it->at(6);	//get lateral distance in frenet coordinates
		if (d < (this->lane_index_ + 1)*this->lane_width_ && d > this->lane_index_*this->lane_width_) {
			double vx = it->at(3);
			double vy = it->at(4);
			double v_abs = sqrt(vx*vx+vy*vy);
			double dist_s = it->at(5);
			dist_s += ((double)this->prev_path_size_*this->delta_t_*v_abs);
			if ((dist_s > this->car_s_) && ((dist_s - this->car_s_) < 30)) {
				this->forward_vehicle_detected = true;
			}
		}
	}
	if (this->forward_vehicle_detected) {
		this->lane_index_ = 0;
	}
	else {
		if (this->ref_vel_ < this->max_vel_) {
			this->ref_vel_ += .2;
		}
	}
}

void pathplanner::speedControl()
{
	//Speed and Distance control to forward vehicle
	double delta_vel = 0.0;
	//Control Speed if ditance larger than 100 meters
	if (this->to_front.s_ > 100) {
		//std::cout << "Speed Control" << std::endl;
		delta_vel = mps2mph(this->max_acc_ * this->delta_t_);
	}
	else {
		//std::cout << "Distance Control" << std::endl;
		double err = this->to_front.s_dot_predicted + (this->to_front.s_predicted - this->car_vel_);
		if (err > this->max_acc_ * this->delta_t_) {
			delta_vel = this->max_acc_ * this->delta_t_;
		}
		else if (err < (this->min_acc_ * this->delta_t_)) {
			delta_vel = this->min_acc_ * this->delta_t_;
		}
		else {
			delta_vel = err;
		}
	}
	if ((this->ref_vel_ + delta_vel) >= this->max_vel_) {
		delta_vel = 0;
	}

	this->ref_vel_ += delta_vel;

}

void pathplanner::debug() {
	/// Objects current relative frenet coordinates
	//std::cout << "Front -- s: " << this->to_front.s_ << "  s_dot: " << this->to_front.s_dot_ << "  s_dot_dot:" << this->to_front.s_dot_dot_ << std::endl;
	//std::cout << "FLeft -- s:" << this->to_front_left.s_ << "s_dot: " << this->to_front_left.s_dot_ << "  s_dot_dot:" << this->to_front_left.s_dot_dot_  << std::endl;
	//std::cout << "BLeft -- s:" << this->to_back_left.s_ << "s_dot: " << this->to_back_left.s_dot_ << "  s_dot_dot:" << this->to_back_left.s_dot_dot_ << std::endl;
	//std::cout << "FRight -- s:" << this->to_front_right.s_ << "s_dot: " << this->to_front_right.s_dot_ << "  s_dot_dot:" << this->to_front_right.s_dot_dot_ << std::endl;
	//std::cout << "BRight -- s:" << this->to_back_right.s_ << "s_dot: " << this->to_back_right.s_dot_ << "  s_dot_dot:" << this->to_back_right.s_dot_dot_ << std::endl;
	/// Objects predicted relative frenet coordinates
	//std::cout << "Front -- predicted -- s: " << this->to_front.s_predicted << "  s_dot: " << this->to_front.s_dot_predicted << std::endl;
	//std::cout << "FLeft -- predicted -- s:" << this->to_front_left.s_predicted << "s_dot: " << this->to_front_left.s_dot_predicted << std::endl;
	//std::cout << "BLeft -- predicted -- s:" << this->to_back_left.s_predicted << "s_dot: " << this->to_back_left.s_dot_predicted << std::endl;
	//std::cout << "FRight -- predicted -- s:" << this->to_front_right.s_predicted << "s_dot: " << this->to_front_right.s_dot_predicted << std::endl;
	//std::cout << "BRight -- predicted -- s:" << this->to_back_right.s_predicted << "s_dot: " << this->to_back_right.s_dot_predicted << std::endl;
	//std::cout << "Previous Path Size:" << this->prev_path_size_ << std::endl;
}
