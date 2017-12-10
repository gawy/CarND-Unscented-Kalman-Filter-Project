#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "measurement_package.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  is_initialized_ = false;
  time_us_ = 0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.3;
  
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

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  //exit if package of type we do not use
  if (!this->use_laser_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    return;
  } else if (!this->use_radar_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
    return;
  }


  //init
  if (!this->is_initialized_) {
    cout<< "Initialising...";
    VectorXd measurement = meas_package.raw_measurements_;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double px = measurement[0] * cos(measurement[1]);
      double py = measurement[0] * sin(measurement[1]);
      this->x_ << px,py,0,0,0;
      cout << "Init with Radar X set to: " << x_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      this->x_ << measurement[0], measurement[1], 0, 0, 0;
      cout << "Init with Laser X set to: " << x_;
    }

    this->time_us_ = meas_package.timestamp_;
    this->is_initialized_ = true;
    cout << "Initilization completed. Time set to: " << this->time_us_ << endl;
    return;
  }

  double delta_t = (meas_package.timestamp_ - this->time_us_) / 1e+6;
  cout << "Time difference: " << delta_t << ", current time: " << meas_package.timestamp_ << endl;
  //Predict
  this->Prediction(delta_t);

  //Update
  if (this->use_laser_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
    this->UpdateLidar(meas_package);
  } else if (this->use_radar_ && meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
    this->UpdateRadar(meas_package);
  }

  this->time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  cout << "Starting prediction magic" << endl;
  this->AugmentedSigmaPoints(&this->Xsig_pred_);
  this->SigmaPointPrediction(&this->Xsig_pred_, delta_t, &this->Xsig_pred_);
  this->PredictMeanAndCovariance(&this->x_, &this->P_);

  cout << "Round of magic accomplished"<< endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  VectorXd z = meas_package.raw_measurements_;
  MatrixXd H = MatrixXd(2, n_x_);
  MatrixXd R = MatrixXd(2, 2);
  R << std_laspx_*std_laspx_, 0,
      0, std_laspy_*std_laspy_;
  H << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0;

  VectorXd y = z - H * x_;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  //new state
  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */


  //set state dimension
  int n_x = this->n_x_;

  //set augmented dimension
  int n_aug = this->n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  int n_sig = 2*n_aug+1;

  //define spreading parameter
  double lambda = this->lambda_;

//set vector for weights
  VectorXd weights = getWeights(n_aug, lambda);

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);


  //transform sigma points into measurement space
  for (int i=0; i < n_sig; i++) {
    Zsig(0,i) = sqrt(Xsig_pred_(0,i)*Xsig_pred_(0,i) + Xsig_pred_(1,i)*Xsig_pred_(1,i)); //ro
    Zsig(1,i) = atan2(Xsig_pred_(1,i), Xsig_pred_(0,i)); //phi
    Zsig(2,i) = (Xsig_pred_(0,i)*cos(Xsig_pred_(3,i))*Xsig_pred_(2,i)
                 + Xsig_pred_(1,i)*sin(Xsig_pred_(3,i))*Xsig_pred_(2,i)) / Zsig(0,i); //ro dot
  }

  //calculate mean predicted measurement
  z_pred.setZero();
  for (int i=0; i<n_sig; i++) {
    z_pred += Zsig.col(i) * weights(i);
  }

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(3,3);
  R(0,0) = std_radr_*std_radr_;
  R(1,1) = std_radphi_*std_radphi_;
  R(2,2) = std_radrd_*std_radrd_;

  S.setZero();
  for (int i=0; i<n_sig; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights(i) * z_diff * z_diff.transpose();
  }
  S += R;

//  cout<< "S:" <<  endl << S << endl;
//  cout<< "z_pred:" <<  endl << z_pred << endl;
//  cout<< "R:" <<  endl << R << endl;


  ///////////////////////// Update state ///////////////////////

  MatrixXd Tc = MatrixXd(n_x, n_z);
  //calculate cross correlation matrix
  Tc.setZero();
  for (int i=0; i<n_sig; i++) {

    VectorXd x_diff = (Xsig_pred_.col(i) - this->x_);
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = (Zsig.col(i) - z_pred);
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc += weights(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  this->x_ = this->x_ + K * (meas_package.raw_measurements_ - z_pred);
  this->P_ = this->P_ - K * S * K.transpose();

  cout << "Updated X:" << endl << x_ << endl;
  cout << "Updated P:" << endl << P_ << endl;

}

VectorXd UKF::getWeights(int n_aug, double lambda) const {
  VectorXd weights = VectorXd(2 * n_aug + 1);
  double weight_0 = lambda/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; i++) {
    double weight = 0.5/(n_aug+lambda);
    weights(i) = weight;
  }
  return weights;
}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  int n_x = this->n_x_;
  int n_aug = this->n_aug_;
  int n_sig = 2 * n_aug + 1;
  double std_a = this->std_a_;
  double std_yawdd = this->std_yawdd_;
  double lambda = this->lambda_;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  P_aug.setZero();

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sig);

  //create augmented mean state
  x_aug.head(n_x) = this->x_;
  x_aug(n_aug - 2) = 0;
  x_aug(n_aug - 1) = 0;

  //create augmented covariance matrix
  P_aug.topLeftCorner(n_x, n_x) = this->P_;

  MatrixXd Q = MatrixXd(2,2);
  Q << std_a*std_a, 0,
      0, std_yawdd*std_yawdd;
  P_aug.bottomRightCorner(2, 2) = Q;
  cout<< "P" << endl << P_ <<endl;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
//  std::cout << "A" << std::endl;
//  std::cout << A << std::endl;

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; i++) {
    //std::cout << i << std::endl;
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * A.col(i);

  }

  //print result
//  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  //write result
  *Xsig_out = Xsig_aug;



}

void UKF::SigmaPointPrediction(MatrixXd *Xsig_aug, double delta_t, MatrixXd* Xsig_out) {
  cout << "Time to see a future in " << delta_t << " seconds" << endl;
  //set state dimension
  int n_x = this->n_x_;
  int n_aug = this->n_aug_;
  int n_sig = 2 * n_aug + 1;

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, n_sig);

  //predict sigma points
  for (int i = 0; i < n_sig; i++) {
    double px = (*Xsig_aug)(0,i);
    double py = (*Xsig_aug)(1,i);
    double v = (*Xsig_aug)(2,i);
    double psi = (*Xsig_aug)(3,i);
    double psi_dot = (*Xsig_aug)(4,i);
    double nu_a = (*Xsig_aug)(5,i);
    double nu_yawdd = (*Xsig_aug)(6,i);


//    double k = psi_dot > 1e-3 ? v/psi_dot : 1;

    //avoid division by zero
    if (fabs(psi_dot) > 0.001) {
      px = px + v/psi_dot * ( sin (psi + psi_dot*delta_t) - sin(psi));
      py = py + v/psi_dot * ( cos(psi) - cos(psi+psi_dot*delta_t) );
    }
    else {
      px = px + v*delta_t*cos(psi);
      py = py + v*delta_t*sin(psi);
    }

    Xsig_pred(0,i) = px
                     + (0.5*delta_t*delta_t * cos(psi) * nu_a);
    Xsig_pred(1,i) = py
                     + (0.5*delta_t*delta_t * sin(psi) * nu_a);
    Xsig_pred(2,i) = v
                     + delta_t * nu_a;
    Xsig_pred(3,i) = psi + psi_dot*delta_t
                     + 0.5*delta_t*delta_t * nu_yawdd;
    Xsig_pred(4,i) = psi_dot
                     + delta_t * nu_yawdd;
  }
  //avoid division by zero
  //write predicted sigma points into right column


  //print result
//  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  *Xsig_out = Xsig_pred;

}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

  //set state dimension
  int n_x = this->n_x_;
  int n_aug = this->n_aug_;
  int n_sig = 2 * n_aug + 1;
  double lambda = this->lambda_;

  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = this->Xsig_pred_;

  //create vector for weights
  VectorXd weights = getWeights(n_aug, lambda);;

  //create vector for predicted state
  VectorXd x = VectorXd(n_x);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);

  //predict state mean
  x.setZero();
  P.setZero();
  for (int i=0; i<n_sig; i++) {
    x += weights(i) * Xsig_pred.col(i);
  }
  //predict state covariance matrix
  for (int i = 0; i < n_sig; i++) {
    VectorXd x_delta = Xsig_pred.col(i) - x;

    //angle normalization
    if ((x_delta(3) < -M_PI) || (M_PI < x_delta(3))) {
//      cout << "Need to normalize x_delta: " << x_delta(3) << endl;

      while (x_delta(3) > M_PI) x_delta(3) -= 2. * M_PI;
      while (x_delta(3) < -M_PI) x_delta(3) += 2. * M_PI;

//      cout << "Normalized to " << x_delta(3) << endl;
    }

//    std::cout << x_delta << std::endl;
//    std::cout << x_delta.transpose() << std::endl;
    P += weights(i) * x_delta * x_delta.transpose();
  }


  //print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P << std::endl;

  //write result
  *x_out = x;
  *P_out = P;
}


