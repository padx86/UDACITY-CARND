#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = .4;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = .7;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  //Initialization state
  is_initialized_ = false;

  //State dimension
  n_x_ = x_.size(); //[px,py,v,yaw,dyaw]

  //Augmented state dimension
  n_aug_ = n_x_+2; //[px,py,v,yaw,dyaw,nu_a,nu_ddyaw]

  //Lambda
  lambda_ = 3 - n_aug_;

  //init nis for lidar and radar
  nis_lidar_ = VectorXd::Zero(2);
  nis_radar_ = VectorXd::Zero(2);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

	if (!this->is_initialized_) {
		//states have not been initialized, dont care which measurement
		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			//Init Lidar
			this->x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;	//Initialize x_
			this->P_ = Eigen::MatrixXd::Identity(this->n_x_, this->n_x_);	//Initialize P_
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			double rho = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];
			this->x_ << rho*cos(phi) , rho*sin(phi), 0.0, 0.0, 0.0;		//Initialize x_
			//Init Radar states
			this->P_ = MatrixXd::Identity(this->n_x_, this->n_x_);	//Initialize P_
		}
		//Initialize timestamp
		this->time_us_ = meas_package.timestamp_;	

		// Allocate Memory for Sigma Points
		this->Xsig_pred_ = MatrixXd::Zero(this->n_x_, 2*this->n_aug_ + 1); 
																			 //Initialize bottom right corner auf P_aug
		this->P_aug_bottom_right_ = MatrixXd(this->n_aug_ - this->n_x_, this->n_aug_ - this->n_x_);
		this->P_aug_bottom_right_ << this->std_a_*this->std_a_, 0, 0, this->std_yawdd_*this->std_yawdd_;
		
		//Initialize weights
		this->weights_ = VectorXd(2 * this->n_aug_ + 1);
		this->weights_(0) = this->lambda_ / (this->lambda_ + this->n_aug_);
		double weight = 1 / (2 * (this->lambda_ + this->n_aug_));
		for (unsigned int i = 1; i < this->weights_.size(); i++) {
			this->weights_(i) = weight;
		}

		//Initialize Lidar measurement Matrix and transpose
		this->H_Lidar_ = MatrixXd(2, this->n_x_);
		this->H_Lidar_ << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
		this->H_Lidar_t_ = this->H_Lidar_.transpose();

		//Initialize Lidar measurement covariance Matrix
		this->R_Lidar_ = MatrixXd(2, 2);
		this->R_Lidar_ << this->std_laspx_*this->std_laspx_, 0.0, 
			0.0, this->std_laspy_*this->std_laspy_;

		//Initialize Radar measurement covariance Matrix
		this->R_Radar_ = MatrixXd(3, 3);
		this->R_Radar_ << this->std_radr_*this->std_radr_, 0.0, 0.0,
			0.0, this->std_radphi_*this->std_radphi_, 0.0,
			0.0, 0.0, this->std_radrd_*this->std_radrd_;

		//Switch init bool
		this->is_initialized_ = true;

		//Output "Init done"
		std::cout << "---------" << std::endl;
		std::cout << "Init done" << std::endl;
		std::cout << "---------" << std::endl;
	}
	else {
		//states have been initialized

		//Predict states
		double dt = (meas_package.timestamp_ - this->time_us_)*1.0e-6; //Calculate delta_t in seconds
		this->time_us_ = meas_package.timestamp_;	//Set time_us_ to current timestamp
		Prediction(dt); //Call prediction function

		//Update measurements
		if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && this->use_laser_) {
			//Update Lidar measurement
			UpdateLidar(meas_package);
		}
		else if((meas_package.sensor_type_ == MeasurementPackage::RADAR) && this->use_radar_) {
			//Update radar measurement
			UpdateRadar(meas_package);
		}
		else {
			//std::cout << "Measurement recieved but no update was done" << std::endl;
		}

		////////////////////
		// outputs /////////
		////////////////////
		//Switch to true to output state and covariance
		if (false) {
			std::cout << "State x_:\n" << this->x_ << endl;
			std::cout << "Covariance P_:\n" << this->P_ << endl;
		}
	}

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
	/////////////////////////////
	//1. Generate Sigma Points //
	/////////////////////////////
	MatrixXd XSig_cur(this->n_aug_, 2 * this->n_aug_ + 1);			//define sigma Points matrix
	MatrixXd P_aug = MatrixXd::Zero(this->n_aug_, this->n_aug_);	//define augmented P matrix
	VectorXd x_aug = VectorXd::Zero(this->n_aug_);					//define augmented state vector
	
    //Fill augmented state
	x_aug.head(5) = this->x_;

	//Fill P_augmented
	P_aug.topLeftCorner(this->n_x_, this->n_x_) = this->P_;
	
	P_aug.bottomRightCorner(this->n_aug_ - this->n_x_, this->n_aug_ - this->n_x_) = this->P_aug_bottom_right_;

	//Create sqrt matrix
	MatrixXd L = P_aug.llt().matrixL();

	//Fill sigma point matrix Xsig_cur
	XSig_cur.col(0) = x_aug;
	double const_dist = sqrt(this->lambda_ + this->n_aug_);
	for (unsigned int i = 1; i <= this->n_aug_; i++) {
		XSig_cur.col(i) = x_aug + const_dist*L.col(i-1);
		XSig_cur.col(i+this->n_aug_) = x_aug - const_dist*L.col(i-1);
	}

	////////////////////////////
	//2. Predict sigma Points //
	////////////////////////////
	double const_noise_pos = .5*delta_t*delta_t;
	double const_noise_rot = .5*delta_t*delta_t;
	for (unsigned int i = 0; i < XSig_cur.cols(); i++) {
		double px0 = XSig_cur(0, i);
		double py0 = XSig_cur(1, i);
		double v0 = XSig_cur(2, i);
		double yaw0 = XSig_cur(3, i);
		double dyaw0 = XSig_cur(4, i);
		double nu_a0 = XSig_cur(5, i);
		double nu_ddyaw0 = XSig_cur(6, i);

		//Predict positions
		double px1 = 0.;
		double py1 = 0.;
		if (fabs(dyaw0) > 1.0e-4) {
			px1 = px0 + (v0 / dyaw0)*(sin(yaw0 + dyaw0*delta_t) - sin(yaw0));
			py1 = py0 + (v0 / dyaw0)*(cos(yaw0) - cos(yaw0 + dyaw0*delta_t));
		}
		else {
			px1 = px0 + v0*cos(yaw0)*delta_t;
			py1 = py0 + v0*sin(yaw0)*delta_t;
		}
		//predict missing states and add noise to position
		px1 = px1 + const_noise_pos*nu_a0*cos(yaw0);
		py1 = py1 + const_noise_pos*nu_a0*sin(yaw0);
		double v1 = v0 + nu_a0*delta_t;
		double yaw1 = yaw0 + dyaw0*delta_t + const_noise_rot*nu_ddyaw0;
		double dyaw1 = dyaw0 + nu_ddyaw0*delta_t;
		this->Xsig_pred_.col(i) << px1, py1, v1, yaw1, dyaw1;
	}

	///////////////////////////////////
	//3. Predict mean and Covariance //
	///////////////////////////////////
	//Predict mean
	this->x_.fill(0.0);
	for (unsigned int i = 0; i < this->Xsig_pred_.cols(); i++) {
		this->x_ = this->x_ + this->weights_(i)*this->Xsig_pred_.col(i);
	}

	//Predict covariance matrix;
	this->P_.fill(0.0);
	for (unsigned int i = 0; i < this->Xsig_pred_.cols(); i++) {
		VectorXd dx = this->Xsig_pred_.col(i) - this->x_;
		NormAngle(dx(3));
		this->P_ = this->P_ + this->weights_(i) * dx * dx.transpose();
	}
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

	//Create and fill predicted vector z_pred
	VectorXd z_pred = this->H_Lidar_*this->x_;
	
	//Create and fill measurement vector z
	VectorXd z = VectorXd(2);
	z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
	
	VectorXd y = z - z_pred;
	MatrixXd S = this->H_Lidar_ * this->P_ * this->H_Lidar_t_ + this->R_Lidar_;
	MatrixXd Si = S.inverse();
	MatrixXd K = this->P_ * this->H_Lidar_t_ * Si;

	this->x_ += (K*y);
	MatrixXd I = MatrixXd::Identity(this->n_x_, this->n_x_);
	this->P_ = (I - (K*this->H_Lidar_))*this->P_;

	// Calculate nis
	double nis = y.transpose()*Si*y;
	// Calculate and output Percent above X.050

	++this->nis_lidar_(0);
	if (nis > 5.991) {
		++this->nis_lidar_(1);
	}
	std::cout << "LIDAR - Percent above X.050 NIS after frame " << this->nis_lidar_(0) << " = " << this->nis_lidar_(1) * 100 / (this->nis_lidar_(0)) << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
	//Get measurement size
	int n_z = meas_package.raw_measurements_.size();

	/////////////////////////////////////////////////////////////
	// 1. Tranform Predicted Sigma Points to measurement space //
	/////////////////////////////////////////////////////////////
	//Allocate memory for measurement sigma points
	MatrixXd Zsig = MatrixXd(n_z, this->Xsig_pred_.cols());

	for (unsigned int i = 0; i < this->Xsig_pred_.cols(); i++) {
		double px1 = this->Xsig_pred_(0, i);
		double py1 = this->Xsig_pred_(1, i);
		double v1 = this->Xsig_pred_(2, i);
		double yaw1 = this->Xsig_pred_(3, i);
		double v1x = v1*cos(yaw1);
		double v1y = v1*sin(yaw1);
		double pxspys = px1*px1 + py1*py1;
		if (pxspys < 1.0e-5) {
			Zsig(0, i) = 0.;
			Zsig(2, i) = 0.;
		}
		else {
			Zsig(0, i) = sqrt(pxspys);
			Zsig(2, i) = (px1*v1x + py1*v1y) / Zsig(0, i);
		}
		Zsig(1, i) = atan2(py1, px1);
	}

	////////////////////////////////////////////////////////////////////////////
	// 2. Predict measurement, covariance matrix and cross correlation matrix //
	////////////////////////////////////////////////////////////////////////////
	//Predict measurement
	VectorXd z_pred = VectorXd::Zero(n_z);
	for (unsigned int i = 0; i < Zsig.cols(); i++) {
		z_pred = z_pred + this->weights_(i)*Zsig.col(i);
	}

	//Covariance matrix S
	MatrixXd S = MatrixXd::Zero(n_z, n_z);
	for (unsigned int i = 0; i < Zsig.cols(); i++) {
		VectorXd dz = Zsig.col(i) - z_pred;
		this->NormAngle(dz(1));
		S = S + this->weights_(i)*dz*dz.transpose();
	}
	S = S + this->R_Radar_;

	//Cross corelation matrix T
	MatrixXd T = MatrixXd::Zero(this->n_x_, n_z);
	for (unsigned int i = 0; i < Zsig.cols(); i++) {
		VectorXd dz = Zsig.col(i) - z_pred;
		this->NormAngle(dz(1));
		VectorXd dx = this->Xsig_pred_.col(i) - this->x_;
		this->NormAngle(dx(3));
		T = T + weights_(i)*dx*dz.transpose();
	}

	////////////////////////////
	// 3. Update Meassurement //
	////////////////////////////
	//Get measurement
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_;

	//Kalman gain
	MatrixXd Si = S.inverse();
	MatrixXd K = T*Si;

	//Get error and update state and covariance
	VectorXd dz = z - z_pred;
	this->NormAngle(dz(1));
	this->x_ += K*dz;
	this->P_ = this->P_ - K*S*K.transpose();

	//Calculate NIS
	double nis = dz.transpose()*Si*dz;
	++this->nis_radar_(0);
	if (nis > 7.815) {
		++this->nis_radar_(1);
	}
	std::cout << "RADAR - Percent above X.050 NIS after frame " << this->nis_radar_(0) << " = " << this->nis_radar_(1) * 100 / (this->nis_radar_(0)) << endl;
}
void UKF::NormAngle(double &angle) {
	//Normalize angles between -PI & PI
	while (angle > M_PI) {
		angle -= 2.*M_PI;
	}
	while (angle < -M_PI) {
		angle += 2.*M_PI;
	}
}
