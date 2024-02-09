// auxiliar_functions.h
#ifndef AUXILIAR_FUNCTIONS_H
#define AUXILIAR_FUNCTIONS_H

#include <Eigen/Dense>
#include <vector>

#include <iostream>
#include "Eigen/LU"
//#include "opencl-c-base.h"
#include "random"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Define a type alias for a collection of matrices
typedef std::vector<Eigen::MatrixXd> matrix_collection;

//Factorial:
unsigned factorial(unsigned n);

// Function to fill vector m with the sum of each column of Z
Eigen::VectorXd fill_m(const MatrixXd& Z);

// Function to count the number of non-zero elements in vector m
unsigned count_nonzero(const Eigen::VectorXd& m);

// Function to eliminate null columns from matrix Z and return a pair of the new matrix and a vector indicating the original non-null columns
std::pair<Eigen::MatrixXd, Eigen::VectorXd> eliminate_null_columns(Eigen::MatrixXd &Z);

// Function to update the precision matrix M given a new row vector z_i
Eigen::MatrixXd update_M(const MatrixXd& M, const VectorXd& z_i);

// Function to calculate the likelihood of the data X given the latent feature matrix Z
long double calculate_likelihood(const MatrixXd& Z, const MatrixXd& X, const MatrixXd& M, double sigma_x,double sigma_a,unsigned K,unsigned D,int n);

// Function to calculate the log-likelihood of the data X given the latent feature matrix Z
//we need it for numerical stability reasons
long double calculate_log_likelihood(const MatrixXd& Z, const MatrixXd& X, 
                                     const MatrixXd& M, double sigma_x, 
                                     double sigma_a, unsigned K, unsigned D, int n);

// Function to calculate the probability of k = certain value in binomial distribution
double binomialProbability(unsigned n_res, double prob, unsigned k);

// Function to calculate the probability of k = certain value in poisson distribution
double poissonProbability(int k, double lambda);

// Metropolis-Hastings step for sigma_a:

double metropolis_step_sigma_a(double current_sigma_a, const MatrixXd& Z, const MatrixXd& X, 
                               const MatrixXd& A, double sigma_x, double proposal_variance,
                               std::default_random_engine& generator, double a_a, double b_a,
                               unsigned K, int& accepted_iterations_a);


// Metropolis-Hastings step for sigma_x:

double metropolis_step_sigma_x(double current_sigma_x, const MatrixXd& Z, const MatrixXd& X, 
                               const MatrixXd& A, double sigma_a, double proposal_variance,
                               std::default_random_engine& generator, double a_x, double b_x,
                               unsigned K, int& accepted_iterations_a);


// Function to sample A matrix
MatrixXd sample_A(const MatrixXd& Z, const MatrixXd& X, double sigma_x, double sigma_a, std::default_random_engine& generator);

//4.2
MatrixXd sample2_A(const MatrixXd& Z, const MatrixXd& X, MatrixXd A, double &sigma_a, double &a, double &b, double &mu_mean, double &mu_var, std::default_random_engine& generator);

// Required for the computation of log[P(X,Z)]
double compute_cardinality(const Eigen::MatrixXd Z);

#endif //AUXILIAR_FUNCTIONS_H
