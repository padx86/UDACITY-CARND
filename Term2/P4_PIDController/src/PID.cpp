#include "PID.h"
#include <iostream>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	//Save controller gains to instance
	this->Kp = Kp;
	this->Kd = Kd;
	this->Ki = Ki;
	this->Ki_init = Ki;
	this->Kd_init = Kd;
	this->Kp_init = Kp;

	//Initialize errors
	this->d_error = 0.;
	this->p_error = 0.;
	this->i_error = 0.;
	
	//Output init message
	std::cout << "PID-controller initialized with Kp="<< Kp << ", Ki=" << Ki <<", Kd=" << Kd << endl;
}

void PID::UpdateError(double cte) {
	if (this->p_error != 0) {
		this->d_error = cte- this->p_error;
	}
	this->p_error = cte;
	this->i_error += this->p_error;
	
}

double PID::TotalError() {
	return this->Kp*this->p_error + this->Kd*this->d_error + this->Ki*this->i_error;
}

void PID::adaptController(double current_speed) {
	if (current_speed > 1.) {
		double factor = .05*current_speed;
		this->Kp = this->Kp_init / factor;
		this->Ki = this->Ki_init / factor;
		this->Kd = this->Kd_init / factor;
	}
}
