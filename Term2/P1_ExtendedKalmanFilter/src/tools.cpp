#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd rmse = VectorXd(4);
	rmse << 0, 0, 0, 0;

	if ((estimations.size() != ground_truth.size()) || estimations.size() == 0) {
		cout << "Invalid estimations or ground truth size" << endl;
		return rmse;
	}
	for (unsigned int i = 0; i < estimations.size(); i++) {
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array()*residual.array();
		rmse += residual;
	}
	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
	MatrixXd jacobian = MatrixXd(3, 4);

	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	if ((px + py) < 0.0001) {
		cout << "divisor (px^2+py^2) near zero --> jacobian will not be calculated" << endl;
		jacobian << 0, 0, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 0;
		return jacobian;
	}

	double pxpy2 = px*px + py*py;
	double pxpy2sr = sqrt(pxpy2);
	double pxpy23sr = pow(pxpy2, 1.5);

	jacobian << px / pxpy2sr, py / pxpy2sr, 0.0, 0.0,
		-py / pxpy2, px / pxpy2, 0.0, 0.0,
		py*(vx*py - vy*px) / pxpy23sr, px*(vy*py - vx*py) / pxpy23sr, px / pxpy2sr, py / pxpy2sr;
	return jacobian;
}
