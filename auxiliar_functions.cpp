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


long double calculate_likelihood(const MatrixXd& Z, const MatrixXd& X, const MatrixXd& M, double sigma_x,double sigma_a,unsigned K,unsigned D,int n) {
  // Assuming a diagonal covariance matrix sigma_X^2 * I for the Gaussian likelihood
  Eigen::MatrixXd Zt=Z.transpose();
    Eigen::MatrixXd Xt=X.transpose();
    //Eigen::MatrixXd I_K=Eigen::MatrixXd::Identity(K,K);
    Eigen::MatrixXd I_n=Eigen::MatrixXd::Identity(n,n);
    long double trace=(Xt *(I_n-(Z * M * Zt ))*X).trace();
    long double det=abs((M*sigma_x*sigma_x).determinant());
    long double den = pow(2*M_PI,n*D/2)*pow(sigma_x,n*D)*pow(sigma_a,K*D);
    return pow(det,D/2)/den*exp(trace*(-1 / (2 * sigma_x * sigma_x)));


}

long double calculate_log_likelihood(const MatrixXd& Z, const MatrixXd& X, 
                                     const MatrixXd& M, double sigma_x, 
                                     double sigma_a, unsigned K, unsigned D, int n) {
  
  // Compute determinant part
  long double log_det_part = 0.5 * D * log(abs((sigma_x*sigma_x*M).determinant()));
  
  // Compute trace part
  MatrixXd mat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * M * Z.transpose())) * X;
  long double trace_part = -0.5 / (sigma_x * sigma_x) * mat.trace();
  
  long double den_part = n*D/2*log(2*M_PI)+n*D*log(sigma_x)+K*D*log(sigma_a);
  
  return log_det_part + trace_part - den_part;
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
   
    for(Eigen::Index i=0;i<Z.cols();++i){
        base_ten = 0;
        for(Eigen::Index j=0; j<Z.rows();++j){
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
            Eigen::Index w = 0;
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



// Funzione per eseguire un passo di Metropolis-Hastings per sigma_x con una prior Inverse Gamma e una proposta lognormale
double metropolis_step_sigma_x(double current_sigma_x, const MatrixXd& Z, const MatrixXd& X, 
                               const MatrixXd& A, double sigma_a, double proposal_variance,
                               std::default_random_engine& generator, double a_x, double b_x, unsigned K, int& accepted_iterations_x) {
  // Trasforma current_sigma_x in eta_x (scala logaritmica)
  double current_eta_x = log(current_sigma_x);
  
  // Genera una proposta per il nuovo eta_x (scala logaritmica)
  std::normal_distribution<double> proposal_dist(current_eta_x, sqrt(proposal_variance));
  double new_eta_x = proposal_dist(generator);
  double new_sigma_x = exp(new_eta_x); // Torna alla scala originale
  
  // Calcola la log-verosimiglianza per sigma_x corrente e proposto
  MatrixXd M_current = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(current_sigma_x/sigma_a, 2)).inverse();
  double current_log_likelihood = calculate_log_likelihood(Z, X, M_current, current_sigma_x,sigma_a, K, A.cols(), Z.rows());
  
  MatrixXd M_new = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(new_sigma_x/sigma_a, 2)).inverse();
  double new_log_likelihood = calculate_log_likelihood(Z, X, M_new, new_sigma_x, sigma_a, K, A.cols(), Z.rows());
  
  // Calcola il log-prior per sigma_x corrente e proposto sotto la distribuzione Inverse Gamma
  double current_log_prior = (a_x * log(b_x) - lgamma(a_x) - (a_x + 1) * log(current_sigma_x) - b_x / current_sigma_x);
  double new_log_prior = (a_x * log(b_x) - lgamma(a_x) - (a_x + 1) * log(new_sigma_x) - b_x / new_sigma_x);
  
  // Includi la correzione Jacobiana nella log-rapporto di accettazione
  double log_acceptance_ratio = std::min(0.0,(new_log_likelihood + new_log_prior + new_eta_x) - (current_log_likelihood + current_log_prior + current_eta_x));
  
  // Decidi se accettare il nuovo valore
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  if (log(uniform(generator)) < log_acceptance_ratio) {
    ++accepted_iterations_x;
    return new_sigma_x;
  } else {
    return current_sigma_x;
  }
}

// Funzione per eseguire un passo di Metropolis-Hastings per sigma_a con una prior Inverse Gamma e una proposta lognormale
double metropolis_step_sigma_a(double current_sigma_a, const MatrixXd& Z, const MatrixXd& X, 
                               const MatrixXd& A, double sigma_x, double proposal_variance,
                               std::default_random_engine& generator, double a_a, double b_a, unsigned K, int& accepted_iterations_a) {
  // Trasforma current_sigma_a in eta_a (scala logaritmica)
  double current_eta_a = log(current_sigma_a);
  
  // Genera una proposta per il nuovo eta_a (scala logaritmica)
  std::normal_distribution<double> proposal_dist(current_eta_a, sqrt(proposal_variance));
  double new_eta_a = proposal_dist(generator);
  double new_sigma_a = exp(new_eta_a); // Torna alla scala originale
  
  // Calcola la log-verosimiglianza per sigma_a corrente e proposto
  MatrixXd M_current = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(sigma_x/current_sigma_a, 2)).inverse();
  double current_log_likelihood = calculate_log_likelihood(Z, X, M_current, sigma_x, current_sigma_a, K, A.cols(), Z.rows());
  
  MatrixXd M_new = (Z.transpose() * Z + MatrixXd::Identity(Z.cols(), Z.cols()) * pow(sigma_x/new_sigma_a, 2)).inverse();
  double new_log_likelihood = calculate_log_likelihood(Z, X, M_new, sigma_x, new_sigma_a, K, A.cols(), Z.rows());
  
  // Calcola il log-prior per sigma_a corrente e proposto sotto la distribuzione Inverse Gamma
  double current_log_prior = (a_a * log(b_a) - lgamma(a_a) - (a_a + 1) * log(current_sigma_a) - b_a / current_sigma_a);
  double new_log_prior = (a_a * log(b_a) - lgamma(a_a) - (a_a + 1) * log(new_sigma_a) - b_a / new_sigma_a);
  
  // Includi la correzione Jacobiana nella log-rapporto di accettazione
  double log_acceptance_ratio = std::min(0.0,(new_log_likelihood + new_log_prior + new_eta_a) - (current_log_likelihood + current_log_prior + current_eta_a));
  
  // Decidi se accettare il nuovo valore
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  
  if (log(uniform(generator)) < log_acceptance_ratio) {
    //std::cout << log_acceptance_ratio<< std::endl;
    ++(accepted_iterations_a);
    //std::cout<< accepted_iterations_a<<std::endl;
    return new_sigma_a;
    
  } else {
    return current_sigma_a;
  }
}


// Function to sample A gaussian matrix (4.1)
MatrixXd sample_A(const MatrixXd& Z, const MatrixXd& X, double sigma_x, double sigma_a, std::default_random_engine& generator) {
  unsigned K = Z.cols(); // Number of features
  unsigned D = X.cols(); // Dimension of data
  
  // Posterior precision and covariance
  MatrixXd posterior_var_inv = (1 / (sigma_x * sigma_x)) * Z.transpose() * Z + (1 / (sigma_a * sigma_a)) * MatrixXd::Identity(K, K);
  //Eigen::LLT<MatrixXd> llt(posterior_var_inv); // Cholesky decomposition for numerical stability
  //MatrixXd posterior_var = llt.solve(MatrixXd::Identity(K, K)); // Invert the precision matrix
  MatrixXd posterior_var = posterior_var_inv.inverse();
  
  // Posterior mean
  //MatrixXd mu_posterior = posterior_var * (Z.transpose() * X) / (sigma_x * sigma_x);
  MatrixXd mu_posterior = (Z.transpose()*Z + (sigma_x*sigma_x/(sigma_a*sigma_a))*MatrixXd::Identity(K,K)).inverse()*Z.transpose()*X;
  
  // Sample from the posterior distribution for A
  MatrixXd new_A(K, D);
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned d = 0; d < D; ++d) {
      std::normal_distribution<double> dist(mu_posterior(k, d), sqrt(posterior_var(k, k)));
      new_A(k, d) = dist(generator);
    }
  }
  return new_A;
}


// Function to sample A matrix with new prior (4.2)
// (a,b): parameters of sigma_a  /  c: constant in the variance of mu_a
MatrixXd sample2_A(const MatrixXd& Z, const MatrixXd& X, MatrixXd A, double &sigma_a, double &a, double &b, double &mu_mean, double &mu_var, std::default_random_engine& generator) {
  unsigned K = Z.cols(); // Number of features  
  unsigned D = X.cols(); // Dimension of data
  double sum1 = 0;
  double sum2 = 0;
  
  // Posterior precision and variance
  a = a + 0.5*K*D; // update a

  for(i=0;i<K;i++) {
      for(j=0;j<D;j++) {
          sum1 = sum1 + std::pow(A[i,j],2);
      }
  }

  double row_mean;
  for(j=0;j<D;j++) {
      row_mean = 0;
      for(i=0;i<K;i++) {
          row_mean = row_mean + A[i,j];
      }
      row_mean = row_mean/K;
      sum2 = sum2 + std::pow(row_mean, 2);
  }
    
  b = b + 0.5*(sum1 + std::pow(K,2)*sum2/(2*(K+1))); // update b
  std::gamma_distribution<double> distr(a, b);  
  double precision = pow(distr(generator),-1); // sto facendo sampling da una gamma
  sigma_a = std::pow(1/precision, 0.5);
  
  // Posterior mean 
  double a_mean = 0;
  for(unsigned cont=0; cont<K; cont++){
    for(unsigned contt=0; contt<D; contt++)
      a_mean += A(cont,contt);
  }
  a_mean = a_mean/(K*D);
  Eigen::VectorXd mu_posterior(K);  
  mu_mean = K*a_mean/(K+1);
  mu_var = mu_var/(K+1);
  for(unsigned k=0; k<K; ++k) {
    std::normal_distribution<double> distr(mu_mean, mu_var);
    mu_posterior(k) = distr(generator);
  }
  
  // Sample from the posterior distribution for A
  MatrixXd new_A(K, D);  
  for(unsigned k=0; k<K; ++k) {
    for(unsigned d=0; d<D; ++d) {
      std::normal_distribution<double> distr(mu_posterior(d), std::pow(sigma_a,2));
      new_A(k,d) = distr(generator); // sampling dei valori di A elemento per elemento    
    }
  }  
  return new_A;
}


long double find_max(VectorXd &v){
    long double max=v(0);
    for (unsigned i=1; i< v.size(); ++i)
        if (v(i) > max)
            max=v(i);
    return max;
}
