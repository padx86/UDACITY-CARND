#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // Define Measurement Matrix
  H_laser_ << 1, 0, 0, 0,
	  0, 1, 0, 0;
  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  noise_ax_ = 30.0;		//Define noise for ekf_.Q_ process covariance matrix
  noise_ay_ = 30.0;		//Define noise for ekf_.Q_ process covariance matrix
  noise_ax_sr_ = 3.0;	//Define noise for ekf_.u_ prediction noise --> currently unused
  noise_ay_sr_ = 3.0;	//Define noise for ekf_.u_ prediction noise --> currently unused

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1.0, 1.0, 1.0, 1.0;

	//first timestamp
	previous_timestamp_ = measurement_pack.timestamp_;

	//Initialize transition matrix ekf_.F_
	ekf_.F_ = MatrixXd::Identity(4,4);

	//Initalize Process covariance matrix ekf_.Q_
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0;
	
	//Initialize State covariance Matrix ekf_.P_
	ekf_.P_ = MatrixXd(4,4);
	ekf_.P_ << 1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 100.0, 0.0,
		0.0, 0.0, 0.0, 100.0;

	//Initialize Noise --> currently unused
	ekf_.u_ = VectorXd(4);
	ekf_.u_ << 0.0, 0.0, 0.0, 0.0;

	VectorXd z = measurement_pack.raw_measurements_;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
		ekf_.x_(0) = measurement_pack.raw_measurements_(0) * cos(measurement_pack.raw_measurements_(1));
		ekf_.x_(1) = measurement_pack.raw_measurements_(0) * sin(measurement_pack.raw_measurements_(1));
		ekf_.x_(2) = 0.0;// measurement_pack.raw_measurements_(2) * cos(measurement_pack.raw_measurements_(1));
		ekf_.x_(3) = 0.0;// measurement_pack.raw_measurements_(2) * sin(measurement_pack.raw_measurements_(1));
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
		ekf_.x_(0) = measurement_pack.raw_measurements_(0);
		ekf_.x_(1) = measurement_pack.raw_measurements_(1);
		ekf_.x_(2) = 0.0;
		ekf_.x_(3) = 0.0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Elapsed time
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  //Update transition Matrix varible part
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //Update process noise
  double dt2 = dt*dt;
  ekf_.u_ << dt2 / 2 * noise_ax_sr_, dt2/2*noise_ay_sr_, dt*noise_ax_sr_, dt*noise_ay_sr_; // Update prediction noise --> currently unused

  //Update process covariance matrix
  double dt3 = dt2*dt;
  double dt4 = dt3*dt;
  dt3 = dt3 / 2.0;
  dt4 = dt4 / 4.0;

  ekf_.Q_(0, 0) = dt4*noise_ax_;
  ekf_.Q_(0, 2) = dt3*noise_ax_;
  ekf_.Q_(1, 1) = dt4*noise_ay_;
  ekf_.Q_(1, 3) = dt3*noise_ay_;
  ekf_.Q_(2, 0) = ekf_.Q_(0, 2);
  ekf_.Q_(2, 2) = dt2*noise_ax_;
  ekf_.Q_(3, 1) = ekf_.Q_(1, 3);
  ekf_.Q_(3, 3) = dt2*noise_ay_;

  ekf_.Predict();
  //cout << "x_pred = " << ekf_.x_ << endl; // Output predictions
  //cout << "P_pred = " << ekf_.P_ << endl; // Output predictions
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	  ekf_.H_ = tools.CalculateJacobian(ekf_.x_);			// Calculate jacobian and set ekf meassurement matrix;
	  ekf_.R_ = R_radar_;									// Set measurement covariance matrix to radar
	  ekf_.UpdateEKF(measurement_pack.raw_measurements_);	// Update radar measurement
  } else {
    // Laser updates
	  ekf_.H_ = H_laser_;									// Set ekf measurement matrix to lidar
	  ekf_.R_ = R_laser_;									// Set measurement covariance matrix to lidar
	  ekf_.Update(measurement_pack.raw_measurements_);		// Update lidar measurement
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
