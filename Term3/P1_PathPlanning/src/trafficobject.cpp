#include "trafficobject.h"

trafficobject::trafficobject(int id, double lane_width, double delta_t) {
	/* Initialize new traffic object with data */
	this->firstdetection = true;
	this->id = id;
	this->lane_width_ = lane_width;
	this->s_dot_ = 0.;
	this->delta_t = delta_t;
	
}

trafficobject::trafficobject() {
	/* Initialize empty traffic object*/
	this->firstdetection = true;
	this->s_ = 1000.;
	this->d_ = 1000.;
	this->lane_index = -10.;
}

int trafficobject::getId()
{
	return this->id;
}

void trafficobject::update(double s, double d, double s_dot, double car_s, double car_d, double car_vel, int frame_count) {
	/* Update meassurement for this traffic object*/
	if (this->firstdetection) {
		//Initialize if first detection
		this->s_dot_dot_ = 0.;
		this->d_dot_ = 0.;
		this->firstdetection = false;
	}
	else {
		//Update if known
		this->s_dot_dot_ = ((s_dot - car_vel) - this->s_dot_) / (this->delta_t*frame_count);
		this->d_dot_ = ((d - car_d) - this->d_) / (this->delta_t*frame_count);
	}
	this->d_ = d - car_d;
	this->s_ = s - car_s;
	this->s_dot_ = s_dot - car_vel;
	this->lane_index = floor(d / this->lane_width_);
	this->relative_lane_index = round((d - car_d) / this->lane_width_);		//Lane index relative to ego vehicle
}


void trafficobject::predict(double prediction_horizon) {
	/* Predict traffic objects kinematics */
	this->s_predicted = this->s_ + this->s_dot_*prediction_horizon + this->s_dot_dot_*pow(prediction_horizon, 2);	//predict longitudinal position with current constant accelaration
	this->s_dot_predicted = this->s_dot_ + this->s_dot_dot_*prediction_horizon;										//predict lonitudinal velocity
	this->d_predicted =this->d_ + this->d_dot_*prediction_horizon;													//predict lane change with current constant lateral velocity
	this->relative_lane_index_predicted = round((this->d_predicted) / this->lane_width_);							//predict traffic objects lane index
}
