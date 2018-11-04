#ifndef TRAFFICOBJECT_H
#define TRAFFICOBJECT_H

#include <vector>
#include <math.h>
using std::vector;

class trafficobject {
private: 
	bool firstdetection;
	double lane_width_;
	double delta_t;
	
	int id;
	
public:
	double s_predicted;
	double s_dot_predicted;
	double d_predicted;
	int relative_lane_index_predicted;

	trafficobject(int id, double lane_width, double delta_t);
	trafficobject();

	double s_;
	double s_dot_;
	double s_dot_dot_;
	double d_;
	double d_dot_;
	int lane_index;
	int relative_lane_index; // - left of; 0 same; + right of;

	int getId();
	void update(double s, double d, double s_dot, double car_s, double car_d, double car_vel, int frame_count);
	void predict(double prediction_horizon);
};
#endif // !TRAFFICOBJECT_H
