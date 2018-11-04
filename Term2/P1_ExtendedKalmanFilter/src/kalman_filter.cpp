#include "kalman_filter.h"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
	x_ = F_ * x_;	// + u_ --> currently unused	//Predict state
	P_ = F_ * P_ * F_.transpose() + Q_;				//Update state covariance matrix
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	
	VectorXd y = z - (H_*x_);								//Calc error
	MatrixXd PHt = P_*H_.transpose();
	MatrixXd S = H_*PHt + R_;					//Calc S matrix
	MatrixXd K = PHt*S.inverse();				//Calc K matrix
	
	x_ = x_ + K * y;										//Update state vector
	int x_size = x_.size();			
	P_ = (MatrixXd::Identity(x_size, x_size) - K*H_)*P_;	//Update covariance matrix
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
	
	VectorXd h = VectorXd(3);
	double pxpy2 = x_(0)*x_(0) + x_(1)*x_(1);								//Calc px^2+py^2
	if(pxpy2 > 0.00001){
		double pxpy2sr = sqrt(pxpy2);												//Calc sqrt(px^2+py^2)
		h << pxpy2sr, atan2(x_(1), x_(0)), (x_(0)*x_(2) + x_(1)*x_(3)) / pxpy2sr;	//Calc h vector
	}
	else {
		h << 0.0, atan2(x_(1), x_(0)), 0;
	}
	while (h(1) - z(1) > M_PI / 2) {
		h(1) -= M_PI;
	}
	while (z(1) - h(1) > M_PI/2) {
		h(1) += M_PI;
	}
	VectorXd y = z - h;											//Calc error
	MatrixXd PHt = P_*H_.transpose();							//Calc P_*H^T
	MatrixXd S = H_*PHt + R_;									//Calc S matrix
	MatrixXd K = PHt*S.inverse();								//Calc K matrix

	x_ = x_ + K * y;														//Update state vector
	int x_size = x_.size();													
	P_ = (MatrixXd::Identity(x_size, x_size) - K*H_)*P_;					//Update covariance matrix

}
