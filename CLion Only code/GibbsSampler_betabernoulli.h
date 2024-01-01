//
// Created by Luca on 31/12/2023.
//

#ifndef BAYESIANA_GIBBSSAMPLER_BETABERNOULLI_H
#define BAYESIANA_GIBBSSAMPLER_BETABERNOULLI_H

#include <Eigen/LU>
#include <list>

std::list<Eigen::MatrixXd> GibbsSampler_betabernoulli( double alpha, double theta, double sigma_x,double sigma_a,  int n_tilde,  int n,  Eigen::MatrixXd A, Eigen::MatrixXd X, unsigned n_iter, unsigned initial_iters);



#endif //BAYESIANA_GIBBSSAMPLER_BETABERNOULLI_H
