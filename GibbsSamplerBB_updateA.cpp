#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include "auxiliar_functions.h" // Include the path to your auxiliary functions

using namespace Rcpp;
using namespace Eigen;

//[[Rcpp::export]]
Rcpp::List GibbsSampler_betabernoulli(double alpha, double theta, double sigma_x, double sigma_a,
                                      double prior_variance_sigma_x, double prior_variance_sigma_a,
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
    
    sigma_x = metropolis_step_sigma(sigma_x, Z, X, A, proposal_variance_factor_sigma_x, generator, true, prior_variance_sigma_x);
    sigma_a = metropolis_step_sigma(sigma_a, Z, X, A, proposal_variance_factor_sigma_a, generator, false, prior_variance_sigma_a);
    
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
      // //VECCHIO METODO per selezionare le new features:
      // // Sample the number of new features and Update Z matrix with new features:
      // double prob = 1-(theta+alpha+n-1)/(theta+n-1);
      // std::binomial_distribution<int> distribution_bin(n_tilde-K,prob);
      // int new_features=distribution_bin(generator);
      // if (new_features > 0) {
      //   Znew = eliminate_null_columns(Z).first;
      // 
      //   // Then, resize Znew and add new columns of ones
      //   Znew.conservativeResize(Eigen::NoChange, Znew.cols() + new_features);
      //   Znew.rightCols(new_features).setZero();
      //   Znew.row(i).tail(new_features).setOnes();
      // 
      //   // Update Z to be Znew
      //   Z = Znew;
      // }
      
      //METODO CON LA POSTERIOR PER SELEZIONARE NUOVE FEATURES(DA FARE QUALCHE CORREZIONE):
      //Vector to store posterior probabilities for each possible number of new features
      std::vector<double> posterior_probabilities(n_tilde - K + 1, 0.0); //utilizza eigen

      MatrixXd Znew = eliminate_null_columns(Z).first;

      for (int new_features = 0; new_features <= n_tilde - K; ++new_features) { //nel caso new_features = 0 va bene scrivere Z = Znew, altrimenti forse no, controlla
        MatrixXd Z_temp = Znew; // Use a temporary matrix to avoid scope issues

        // Resize Z_temp and add new columns of ones for each potential new feature
        Z_temp.conservativeResize(Eigen::NoChange, Z_temp.cols() + new_features); //potrebbe dare problemi
        Z_temp.rightCols(new_features).setZero();
        Z_temp.block(0, Z_temp.cols() - new_features, 1, new_features).setOnes(); // Set the last 'new_features' rows of the ith row to ones

        // Update M matrix for Z_temp
        MatrixXd M_temp = (Z_temp.transpose() * Z_temp + MatrixXd::Identity(Z_temp.cols(), Z_temp.cols()) * pow(sigma_x / sigma_a, 2)).inverse();

        // Calculate likelihood with new features
        long double likelihood_new_features = calculate_likelihood(Z_temp, X, M_temp, sigma_x, sigma_a, n_tilde, D, n);

        // Calculate the posterior probability for this number of new features
        double prior_probability = (theta + alpha) / (theta + n);
        double success_probability = 1.0 - prior_probability;

        if (success_probability >= 0.0 && success_probability <= 1.0) {
          std::binomial_distribution<int> binomial_dist(new_features, success_probability);
          double prior_prob = binomial_dist(generator);
          posterior_probabilities[new_features] = likelihood_new_features * prior_prob;
        } else {
          Rcpp::Rcerr << "Invalid probability for binomial distribution: " << success_probability << std::endl;
          return Rcpp::List::create(Rcpp::Named("error") = "Invalid probability for binomial distribution");
        }
      }

      // Normalize posterior probabilities
      double sum_posterior = std::accumulate(posterior_probabilities.begin(), posterior_probabilities.end(), 0.0);
      for (double& prob : posterior_probabilities) {
        prob /= sum_posterior;
      }

      // Sample the number of new features based on posterior probabilities
      std::discrete_distribution<int> distribution(posterior_probabilities.begin(), posterior_probabilities.end());
      int sampled_new_features = distribution(generator);

      // Add the sampled number of new features to Znew
      Znew.conservativeResize(Eigen::NoChange, Znew.cols() + sampled_new_features);
      Znew.rightCols(sampled_new_features).setZero();
      Znew.block(0, Znew.cols() - sampled_new_features, 1, sampled_new_features).setOnes(); // Set the last 'sampled_new_features' rows of the ith row to ones

      // Update Z to be Znew
      Z = Znew; //attento a sta roba
     }
    
    // Store the Z matrix with non-null columns only after initial iterations
    if (it >= initial_iters) {
      Ret.push_back(eliminate_null_columns(Z).first);
    }
    for (Eigen::Index i = 0; i < Z.rows(); ++i) {
      for (Eigen::Index j = 0; j < Z.cols(); ++j) {
        std::cout << Z(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }
  
  // Eliminate null columns from Z if necessary (POI CONTROLLA SE E' NECESSARIO E SE LE DIMENSIONI FUNZIONANO!)
  std::pair<MatrixXd, VectorXd> Z_pair = eliminate_null_columns(Z);
  Z = Z_pair.first; // Z with null columns eliminated
  
  // Update the A matrix using the sample_A function
  MatrixXd A_updated = sample_A(Z, X, sigma_x, sigma_a, generator);
  
  
  // Return the result as a List containing the matrices and K_vector
  return Rcpp::List::create(Rcpp::Named("result") = Ret, Rcpp::Named("K_vector") = K_vector);

}

