#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include "auxiliar_functions.h" // Include the path to your auxiliary functions

using namespace Rcpp;
using namespace Eigen;

//[[Rcpp::export]]
Rcpp::List GibbsSampler_betabernoulli(double alpha, double theta, double sigma_x, double sigma_a,
                                      int n_tilde, int n, const Eigen::Map<Eigen::MatrixXd>& A,
                                      const Eigen::Map<Eigen::MatrixXd>& X, unsigned n_iter,
                                      unsigned initial_iters) {
  std::default_random_engine generator;
  MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde); // Initialize Z matrix
  VectorXd m(n_tilde); // Initialize m vector
  unsigned D = A.cols(); // Number of properties
  matrix_collection Ret; // Collection to store matrices
  VectorXd K_vector(n_iter + initial_iters); // Vector to store the number of features (number of features that are active at each iteration of the sampler)
  
  for (unsigned it = 0; it < n_iter + initial_iters; ++it) {
    // Update sigma_x and sigma_a using Metropolis-Hastings steps
    double proposal_variance_factor_sigma_x = 0.1 * sigma_x; // e.g., 10% of current sigma_x
    double proposal_variance_factor_sigma_a = 0.1 * sigma_a; // e.g., 10% of current sigma_a
    
    sigma_x = metropolis_step_sigma(sigma_x, Z, X, A, proposal_variance_factor_sigma_x, generator, true);
    sigma_a = metropolis_step_sigma(sigma_a, Z, X, A, proposal_variance_factor_sigma_a, generator, false);
    
    MatrixXd Znew;
    
    // Initialize M matrix
    MatrixXd M = (Z.transpose() * Z + MatrixXd::Identity(n_tilde, n_tilde) * pow(sigma_x/sigma_a, 2)).inverse();
    
    for (int i = 0; i < n; ++i) {
      VectorXd z_i = Z.row(i);
      Z.row(i).setZero();
      
      // Eliminate null columns from Z and get the new Z matrix along with the positions of non-null columns
      std::pair<MatrixXd, VectorXd> matvec = eliminate_null_columns(Z);
      Znew = matvec.first;
      VectorXd positions = matvec.second;
      
      Z.row(i) = z_i; // Set the row back after eliminating null columns
      m = fill_m(Znew); // Fill m with the sum of each column of Znew
      
      // Update the number of observed features
      unsigned K = count_nonzero(m);
      K_vector(it) = K; // Store the number of features
      
      Eigen::Index count = 0; // Index for mapping Znew to Z
      
      for (int j = 0; j < K; ++j) {
        while (positions(count) == 0) {
          ++count;
        }
      
        // Calculate the probabilities and sample from Bernoulli distribution
        double prior_1 = (m(j) - alpha) / (theta + (n - 1));
        
        // Calculate the likelihood when Z(i, j) = 1
        Z(i, count) = 1;
        M = update_M(M, Z.row(i));
        long double likelihood_1 = calculate_likelihood(Z, X, M, sigma_x,sigma_a,n_tilde,D,n);
        
        // Calculate the likelihood when Z(i, j) = 0
        Z(i, count) = 0;
        M = update_M(M, Z.row(i));
        long double likelihood_0 = calculate_likelihood(Z, X, M, sigma_x,sigma_a,n_tilde,D,n);
        
        // Combine the likelihood and prior to get the full conditional probability
        double prob_1 = prior_1 * likelihood_1 / (prior_1 * likelihood_1 + (1 - prior_1) * likelihood_0);
        //Rcpp::Rcout << "Prior on the parameter of the bern: " <<  prob_1 << std::endl;
        std::bernoulli_distribution distribution(prob_1);
        Z(i, count) = distribution(generator); // Sample the new value of Z(i, j) from the Bernoulli distribution
      }
       //sample the number of new features:



            unsigned n_res = n_tilde - K;
            if(n_res>0){
                //sample the number of new features:

                //update Z-part1:
                Eigen::Index j = 0;
                for (; j < Znew.cols(); ++j)
                    Z.col(j) = Znew.col(j);
                for (Eigen::Index kk = j; kk < Z.cols(); ++kk)
                    Z.col(kk).setZero();


                M = (Z.transpose() * Z -
                     Eigen::MatrixXd::Identity(n_tilde, n_tilde) * pow(sigma_x / sigma_a, 2)).inverse();


                double prob = 1 - (theta + alpha + n - 1) / (theta + n - 1);

                Eigen::VectorXd prob_new(n_res);
                for (unsigned itt = 0; itt < n_res; ++itt) {
                    double bin_prob = binomialProbability(n_res,prob, itt);
                    Z(i, j+itt) = 1;
                    M = update_M(M, Z.row(i));
                    long double p_xz_1 = 1 / (pow((Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *
                                                                       Eigen::MatrixXd::Identity(n_tilde,
                                                                                                 n_tilde)).determinant(),
                                                  D * 0.5));
                    MatrixXd mat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * M * Z.transpose())) * X;
                    long double p_xz_2 = mat.trace() * (-1 / (2 * sigma_x * sigma_x));

                    double p_xz = p_xz_1 * exp(p_xz_2);
                    prob_new(itt) = bin_prob * p_xz;
                }
                // Normalize posterior probabilities
                double sum_posterior = prob_new.sum()
                for (unsigned l=0;l<prob_new.size();++l) {
                        prob_new(l) /= sum_posterior;
                      }
        
               // Sample the number of new features based on posterior probabilities
               std::discrete_distribution<int> distribution(prob_new.begin(), prob_new.end());
               int new_feat = distribution(generator);


                //update Z-part2:
                j--;
                for (; j >= Znew.cols() + new_feat; --j) {
                    Z(i, j) = 0;
                }
            }

            else
                Z=Znew;
            
            
            }

        if(it>=initial_iters){
            Ret.push_back(eliminate_null_columns(Z).first);

        }

    }


 // Return the result as a List containing the matrices and K_vector
  return Rcpp::List::create(Rcpp::Named("result") = Ret, Rcpp::Named("K_vector") = K_vector);

}
