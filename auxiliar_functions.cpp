// auxiliar_functions.cpp
#include "auxiliar_functions.h"
#include <algorithm>

// Function to fill vector m with the sum of each column of Z
Eigen::VectorXd fill_m(const MatrixXd& Z) {
  Eigen::VectorXd m(Z.cols());
  for(Eigen::Index i = 0; i < Z.cols(); ++i) {
    m(i) = Z.col(i).sum();
  }
  return m;
}

// Function to count the number of non-zero elements in vector m
unsigned count_nonzero(const Eigen::VectorXd& m) {
  // Initialize a counter for non-zero elements
  unsigned count = 0;
  
  // Loop over all elements in the vector
  for (int i = 0; i < m.size(); ++i) {
    // If the element is not zero, increment the counter
    if (m[i] > 0) {
      ++count;
    }
  }
  
  return count;
}


// Function to eliminate null columns from matrix Z and return a pair of the new matrix and a vector indicating the original non-null columns
std::pair<Eigen::MatrixXd, Eigen::VectorXd> eliminate_null_columns(Eigen::MatrixXd &Z) {
  Eigen::Index cols = Z.cols();
  Eigen::VectorXd m=fill_m(Z);
  size_t colnew=count_nonzero(m);
  MatrixXd Ret(Z.rows(), colnew);
  Ret.setZero();
  VectorXd ret(cols);
  ret.setZero();
  size_t iter=0;
  
  for (Eigen::Index i = 0; i < cols &&iter<colnew; ++i) {
    if (Z.col(i).norm() > 0) {
      Ret.col(iter)=Z.col(i);
      ret(i)=1;
      ++iter;
    }
    
  }
  
  return std::make_pair(Ret,ret);
}

// Function to update the precision matrix M given a new row vector z_i
Eigen::MatrixXd update_M(const MatrixXd& M, const VectorXd& z_i) {
  MatrixXd M_i= M - (M*z_i*z_i.transpose()*M)/((z_i.transpose()*M*z_i) - 1);
  MatrixXd M_mod=M_i - (M_i*z_i*z_i.transpose()*M_i)/((z_i.transpose()*M_i*z_i) +1 );
  return M_mod;
}


long double calculate_likelihood(const MatrixXd& Z, const MatrixXd& X, const MatrixXd& M, double sigma_x,double sigma_a,int n_tilde,unsigned D,int n) {
  // Assuming a diagonal covariance matrix sigma_X^2 * I for the Gaussian likelihood
  long double prob_xz_1=1/(pow((Z.transpose()*Z + sigma_x*sigma_x/sigma_a/sigma_a*Eigen::MatrixXd::Identity(n_tilde,n_tilde)).determinant(),D*0.5) );
  MatrixXd mat=X.transpose() * (Eigen::MatrixXd::Identity(n,n) - (Z * M * Z.transpose())) * X;
  long double prob_xz_2=mat.trace()*(-1/(2*sigma_x*sigma_x));
  return prob_xz_1*exp(prob_xz_2);
}

// Function to calculate the probability of k = certain value in binomial distribution
double binomialProbability(unsigned n_res, double prob, unsigned k) {
    double binomial_coefficient = factorial(n_res) / (factorial(k) * factorial(n_res - k));
    return binomial_coefficient * pow(prob, k) * pow(1 - prob, n_res - k);
}

double poissonProbability(int k, double lambda) {
    return (exp(-lambda) * pow(lambda, k)) / tgamma(k + 1);
}

// Function to perform a Metropolis-Hastings step for sigma_x or sigma_a
double metropolis_step_sigma(double current_sigma, const MatrixXd& Z, const MatrixXd& X, 
                             const MatrixXd& A, double proposal_variance, std::default_random_engine& generator, 
                             bool is_sigma_x,double prior_variance) {
  std::normal_distribution<double> proposal_dist(current_sigma, sqrt(proposal_variance));
  double new_sigma = proposal_dist(generator);
  
  if (new_sigma <= 0) return current_sigma; // Ensure proposed sigma is positive
  
  double current_log_likelihood, new_log_likelihood;;
  
  if (is_sigma_x) {
    // Calculate log likelihood with the current and new sigma_x
    MatrixXd M = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(current_sigma / sigma_a, 2)).inverse();
    current_log_likelihood = calculate_log_likelihood(Z, X, M, current_sigma, sigma_a, Z.cols(), A.cols(), Z.rows());
    
    M = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(new_sigma / sigma_a, 2)).inverse();
    new_log_likelihood = calculate_log_likelihood(Z, X, M, new_sigma, sigma_a, Z.cols(), A.cols(), Z.rows());
  } else {
    // Calculate log likelihood with the current and new sigma_a
    MatrixXd M = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(sigma_x / current_sigma, 2)).inverse();
    current_log_likelihood = calculate_log_likelihood(Z, X, M, sigma_x, current_sigma, Z.cols(), A.cols(), Z.rows());
    
    M = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(sigma_x / new_sigma, 2)).inverse();
    new_log_likelihood = calculate_log_likelihood(Z, X, M, sigma_x, new_sigma, Z.cols(), A.cols(), Z.rows());
  }
  
  // Calculate log-prior for current and proposed sigma
  double current_log_prior = -0.5 * current_sigma * current_sigma / prior_variance;
  double new_log_prior = -0.5 * new_sigma * new_sigma / prior_variance;
  
  // Compute the log of the acceptance ratio
  double log_acceptance_ratio = (new_log_likelihood + new_log_prior) - (current_log_likelihood + current_log_prior);
  
  // Decide whether to accept the new value
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  if (log(uniform(generator)) < log_acceptance_ratio) {
    return new_sigma; // Accept the new value
  } else {
    return current_sigma; // Reject the new value
  }
}



