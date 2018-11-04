#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_; //done

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_; //done

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_; //done

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_; //done

  ///* state covariance matrix
  MatrixXd P_; //done

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_; //done

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_; //done --> requires changes

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_; //done -->reqires changes

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_; //done

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_; //done

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_; //done

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_; //done

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ; //done

  ///* Weights of sigma points
  VectorXd weights_; //done

  ///* State dimension
  int n_x_; //done

  ///* Augmented state dimension
  int n_aug_; //done

  ///* Sigma point spreading parameter
  double lambda_; //done

  ///* Bottom right Matrix for P_augmented
  MatrixXd P_aug_bottom_right_; //done

  ///* Lidar measurement matrix
  MatrixXd H_Lidar_; //done

  ///* Lidar measurement matrix transposed
  MatrixXd H_Lidar_t_; //done

  ///* Lidar measurement covariance matrix
  MatrixXd R_Lidar_; //done

  ///* Radar measurement covariance matrix
  MatrixXd R_Radar_; //done

  ///* NIS Vector for Radar
  VectorXd nis_lidar_; //done

  ///*NIS Vector for Lidar
  VectorXd nis_radar_; //done
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  /**
  * Normalizes Angle between -M_PI & M_PI
  * @param angle to be normalized
  */
  void NormAngle(double &angle);
};

#endif /* UKF_H */
