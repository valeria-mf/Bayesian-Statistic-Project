// auxiliar_functions.cpp
#include "auxiliar_functions.h"
#include <algorithm>


//Factorial:
unsigned factorial(unsigned n){
    if (n==0 || n==1)
        return 1;
    return n * factorial(n-1);
}

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


// Per il calcolo delle log[P(X|Z)]
double compute_cardinality(Eigen::MatrixXd Z) {
    
    int base_ten;
    int k=0;
    Eigen::VectorXd histories = Eigen::VectorXd::Constant(Z.cols(), -1);
    Eigen::VectorXd occurrences = Eigen::VectorXd::Constant(Z.cols(), 0); // con una unordered map sarebbe molto meglio, ma ad Eigen non piacciono gli iteratori :(
   
    for(size_t i=0;i<Z.cols();++i){
        base_ten = 0;
        for(size_t j=0; j<Z.rows();++j){
            if(Z(j,i)==1){
                base_ten+=pow(2,Z.rows()-j-1); // la colonne di Z vengono lette in binario dall'alto in basso e trasformata in base 10
                // lo faccio per identificare univocamente e facilmente features con la stessa storia (colonne uguali => base_ten uguali)
                // potrà anche tornare utile per creare la left_order_form (dove potremo ordinare le colonne da sx a dx in ordine decrescente di numero base_ten)
            }
        }
/*
        Rcpp::cout << "base_ten da inserire: " << base_ten << endl;
        Rcpp::cout << "queste sono le histories fino ad ora: " << histories << endl;
        Rcpp::cout << "E queste le loro occurrences : " << occurrences << endl;
*/
        for(k=0;k<occurrences.size();++k){
            if(histories(k)==base_ten){            // se la base_10 della i-esima colonna di Z è già stata messa in histories
                occurrences(k)+=1;                 // allora ne aumento il contatore di occorrenze
                break;
            }
        }

        if(k == occurrences.size()){               // alternativamente, avremo trovato una nuova history...
            size_t w = 0;
            while(occurrences(w) != 0 && w < occurrences.size()){
                w=w+1;
            }
            histories(w) = base_ten;                // che cataloghiamo nel primo slot libero di histories
            occurrences(w) = 1;                     // con contatore di occorrenze in occurrences inizializzato ad 1
        }
    }

    double log_cardinality = 0;                        
    int current_occurrence = 0;                   
    int q = 1;

    //Rcpp::cout << "occurrences: " << occurrences << endl;
    for(int p=Z.cols();p>0;p--){
        if(q>occurrences(current_occurrence)){
            q = 1;
            current_occurrence++;
        }
        //Rcpp::cout << "log-cardinality is: " << log_cardinality << endl;
        //Rcpp::cout << "p/q: " << p << "/" << q << endl;
        log_cardinality = log_cardinality+log(p)-log(q);
        q++;
    }

    //Rcpp::cout << "cardinality is: " << log_cardinality << endl;
    return log_cardinality;
}


/*
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

// Function to sample A gaussian matrix (4.1)
MatrixXd sample_A(const MatrixXd& Z, const MatrixXd& X, double sigma_x, double sigma_a, std::default_random_engine& generator) {
  unsigned K = Z.cols(); // Number of features
  unsigned D = X.cols(); // Dimension of data
  
  // Posterior precision and covariance
  MatrixXd Sigma_posterior_inv = (1 / (sigma_x * sigma_x)) * Z.transpose() * Z + (1 / (sigma_a * sigma_a)) * MatrixXd::Identity(K, K);
  LLT<MatrixXd> llt(Sigma_posterior_inv); // Cholesky decomposition for numerical stability
  MatrixXd Sigma_posterior = llt.solve(MatrixXd::Identity(K, K)); // Invert the precision matrix
  
  // Posterior mean
  MatrixXd mu_posterior = Sigma_posterior * (Z.transpose() * X) / (sigma_x * sigma_x); // formula usata da antonio
  MatrixXd mu_posterior = (Z.transpose() * Z + sigma_x^2 * MatrixXd::Identity(Z.cols(),Z.cols()) / sigma_a^2).inv() * Z.transpose() * X; // formula secondo me (giuseppe)
  
  // Sample from the posterior distribution for A
  MatrixXd new_A(K, D);
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned d = 0; d < D; ++d) {
      std::normal_distribution<double> dist(mu_posterior(k, d), sqrt(Sigma_posterior(k, k)));
      new_A(k, d) = dist(generator);
    }
  }
  
  return new_A;
}




// Function to sample A matrix with new prior (4.2)
MatrixXd sample2_A(const MatrixXd& Z, const MatrixXd& X, double sigma_x, double sigma_a, std::default_random_engine& generator) {
  unsigned K = Z.cols(); // Number of features
  unsigned D = X.cols(); // Dimension of data
  
  // Posterior precision and covariance
  a = 1; // per ora è un numero a caso
  b = 1; // idem
  std::gamma_distribution<double> distr(a, b) // bisogna capire cosa mettere come a e b
  double Sigma_posterior = pow(distr(generator),-1) // sto facendo sampling da una gamma
  
  
  // Posterior mean
  c = 1 // per ora è un numero a caso
  Eigen::VectorXd mu_posterior(D);
  for(unsigned d=0; d<D; ++d) {
    std::normal_distribution<double> distr(0, c*Sigma_posterior); // sampling dei valori della media elemento per elemento
    mu_posterior(d) = distr(generator);
  }

  
  // Sample from the posterior distribution for A
  MatrixXd new_A(K, D);
  for(unsigned k=0; k<K; ++k) {
    for(unsigned d=0; d<D; ++d) {
      std::normal_distribution<double> distr(mu_posterior(d), Sigma_posterior);
      new_A(k,d) = distr(generator); // sampling dei valori di A elemento per elemento
    }
  }
  return new_A;
  }
  // Fine 4.2
*/
