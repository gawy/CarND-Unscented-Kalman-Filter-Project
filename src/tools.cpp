#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd d;
  VectorXd rmse = VectorXd::Zero(estimations[0].rows(), estimations[0].cols());
  for (int i = 0; i < estimations.size(); i++) {
    d = estimations[i] - ground_truth[i];
    rmse = rmse + d.cwiseAbs2();
  }
  rmse = rmse / estimations.size();
  rmse = rmse.cwiseSqrt();

  return rmse;

}