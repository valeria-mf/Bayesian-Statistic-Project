// auxiliar_functions.cpp
#include "auxiliar_functions.h"
#include <algorithm>
#include <cmath>
#define pi 3.1415926535897932

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
double metropolis_step_sigma_a(double current_sigma, const MatrixXd& Z, const MatrixXd& X,
                               const MatrixXd& A, std::default_random_engine& generator,double sd) {

    unsigned K=A.rows(), D=A.cols();
    int seed=29;
    double new_sigma= truncated_normal_a_sample(current_sigma,sd,0.0,seed);

    double ratio= pow(new_sigma/current_sigma,K*D)*exp(-1/2*(A.transpose()*A).trace()*(1/pow(new_sigma,2)-1/pow(current_sigma,2)))*truncated_normal_a_pdf(current_sigma,new_sigma,sd,0)/truncated_normal_a_pdf(new_sigma,current_sigma,sd,0);
    double alpha = ratio>1? 1: ratio;
    std::bernoulli_distribution distribution_bern(alpha);

    return distribution_bern(generator) ? new_sigma : current_sigma;


}

double metropolis_step_sigma_x(double current_sigma, const MatrixXd& Z, const MatrixXd& X,
                               const MatrixXd& A, std::default_random_engine& generator,double sd){
    unsigned N=Z.rows(), D=A.cols();
    int seed=29;
    double new_sigma= truncated_normal_a_sample(current_sigma,sd,0.0,seed);



    double ratio= pow(current_sigma/new_sigma,N*D)
            *exp(-0.5*((X-(Z*A))*((X-(Z*A)).transpose())).trace())
            *(1/pow(new_sigma,2)-1/pow(current_sigma,2))
            *truncated_normal_a_pdf(current_sigma,new_sigma,sd,0)/truncated_normal_a_pdf(new_sigma,current_sigma,sd,0);
    double alpha = ratio>1? 1: ratio;

    std::bernoulli_distribution distribution_bern(alpha);

    return distribution_bern(generator) ? new_sigma : current_sigma;

}



unsigned factorial(unsigned n){
    if (n==0 || n==1)
        return 1;
    return n* factorial(n-1);
}

double normal_pdf(double x, double mean, double stddev) {
    double coefficient = 1.0 / (stddev * sqrt(2 * M_PI));
    double exponent = exp(-0.5 * pow((x - mean) / stddev, 2));
    return coefficient * exponent;
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
        cout << "base_ten da inserire: " << base_ten << endl;
        cout << "queste sono le histories fino ad ora: " << histories << endl;
        cout << "E queste le loro occurrences : " << occurrences << endl;
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

    //cout << "occurrences: " << occurrences << endl;
    for(int p=Z.cols();p>0;p--){
        if(q>occurrences(current_occurrence)){
            q = 1;
            current_occurrence++;
        }
        //cout << "log-cardinality is: " << log_cardinality << endl;
        //cout << "p/q: " << p << "/" << q << endl;
        log_cardinality += log(p)-log(q);
        q++;
    }

    //cout << "cardinality is: " << log_cardinality << endl;
    return log_cardinality;
}





// QUI I GIBBS SAMPLRES ------------------------------------------------------------------------------------------------





matrix_collection gibbsSamplerBetabernoulli( double alpha, double theta, double sigma_x,double sigma_a,  int n_tilde,  int n,  MatrixXd &A, MatrixXd &X, unsigned n_iter, unsigned initial_iters) {

    /*STRATEGY:
     * When generatig a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices will be inserted matrices Z with only non null columns.
  */
    std::default_random_engine generator;


    // Initialization of Z and m:
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);

    std::bernoulli_distribution Z_initializer(0.5);
    for(unsigned i=0; i< n ; ++i)
        for(unsigned j=0; j<n_tilde;++j)
            Z(i, j) = Z_initializer(generator) ? 1 : 0;
    //  std::cout<< Z << std::endl;


    VectorXd m(n_tilde);

    // D:
    const unsigned D = A.cols();

    //create a set to put the generated Z matrices:
    matrix_collection Ret;





    for (Eigen::Index it=0;it<n_iter+initial_iters;++it){

        double proposal_variance_factor_sigma_x = 0.1 * sigma_x; // e.g., 10% of current sigma_x
        double proposal_variance_factor_sigma_a = 0.1 * sigma_a; // e.g., 10% of current sigma_a

        sigma_x = metropolis_step_sigma_x(sigma_x,Z,X,A,generator,proposal_variance_factor_sigma_x);
        sigma_a = metropolis_step_sigma_a(sigma_a,Z,X,A,generator,proposal_variance_factor_sigma_a);



        MatrixXd Znew;

        //INITIALIZE M MATRIX:
        MatrixXd M=(Z.transpose()*Z -  Eigen::MatrixXd::Identity(n_tilde,n_tilde)*pow(sigma_x/sigma_a,2)).inverse();


        for (Eigen::Index i=0; i<n;++i) {

            Eigen::VectorXd z_i = Z.row(i);


            Z.row(i).setZero();

            VectorXd positions;
            auto matvec = std::make_pair(Znew, positions);
            matvec = eliminate_null_columns(Z);
            Znew = matvec.first; //the new matrix that I will update
            positions = matvec.second; //to see the positions where I remove the columns

            Z.row(i) = z_i;
            m = fill_m(Znew);

            //update the number of observed features:
            unsigned K = count_nonzero(m);

            Eigen::Index count = 0;

            for (Eigen::Index j = 0; j < K; ++j) {
                while (positions(count) == 0)
                    ++count;

                double prob_zz = (m(j) - alpha) / (theta + (n - 1));

                //P(X|Z) when z_ij=1:
                Z(i, count) = 1;
                M = update_M(M, Z.row(i));

                long double prob_xz_1 = 1 / (pow((Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *
                                                                      Eigen::MatrixXd::Identity(n_tilde,
                                                                                                n_tilde)).determinant(),
                                                 D * 0.5));
                MatrixXd mat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * M * Z.transpose())) * X;
                long double prob_xz_2 = mat.trace() * (-1 / (2 * sigma_x * sigma_x));


                long double prob_xz = prob_xz_1 * exp(prob_xz_2);


                //P(X|Z) when z_ij=0:
                Z(i, count) = 0;
                M = update_M(M, Z.row(i));
                long double prob_xz_10 = 1 / pow((Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *
                                                                      Eigen::MatrixXd::Identity(Z.cols(),
                                                                                                Z.cols())).determinant(),
                                                 D * 0.5);

                MatrixXd mat0 = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * M * Z.transpose())) * X;
                long double prob_xz_20 = mat.trace() * (-1 / (2 * sigma_x * sigma_x));

                long double prob_xz0 = prob_xz_10 * exp(prob_xz_20);

                //Bernoulli parameter:

                long double prob_one_temp = prob_zz * prob_xz;
                long double prob_zero_temp = (1 - prob_zz) * prob_xz0;
                long double prob_param = prob_one_temp / (prob_one_temp + prob_zero_temp); //PROBLEM: always too small

                //sample from Bernoulli distribution:
                std::bernoulli_distribution distribution_bern(prob_param);

                Znew(i, j) = distribution_bern(generator) ? 1 : 0;
                Z(i, count) = Znew(i, j);


                ++count;
            }

            unsigned n_res = n_tilde - K;
            if (n_res > 0) {
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
                    double bin_prob = binomialProbability(n_res, prob, itt);
                    Z(i, j + itt) = 1;
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
                double sum_posterior = prob_new.sum();
                for (unsigned l = 0; l < prob_new.size(); ++l) {
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
        }


        //Alla fine di ogni iterazione calcolo la quantità log[P(X|Z)]
        //----------------------------------------------------------------------

        std::cout << Z << std::endl;
        std::cout << " " << std::endl;
        //Per il BB utilizzo Eq 21 e 12:

        int K = A.rows();
        int D = A.cols();

        // Eq 21 dopo averla messa nel logaritmo:

        long double eq_21_log_denominator =
                -(n * D / 2) * log(2 * pi) - (n - K) * D * log(sigma_x) - K * D * log(sigma_a) - D / 2 *
                                                                                                 log((Z.transpose() *
                                                                                                      Z + sigma_x *
                                                                                                          sigma_x /
                                                                                                          sigma_a /
                                                                                                          sigma_a *
                                                                                                          Eigen::MatrixXd::Identity(
                                                                                                                  Z.cols(),
                                                                                                                  Z.cols())).determinant());                                                                                    //

        Eigen::MatrixXd MM = (Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *
                                                  Eigen::MatrixXd::Identity(Z.cols(), Z.cols())).inverse();

        MatrixXd matmat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * MM * Z.transpose())) * X;
        long double eq_21_log_exponential = -matmat.trace() / sigma_x / sigma_x / 2;
        long double eq_21_log = eq_21_log_denominator + eq_21_log_exponential;
        //std::cout << "eq_21_log = eq_21_log_denominator + eq_21_log_exponential = " << eq_21_log_denominator << " + " << eq_21_log_exponential << " = " << eq_21_log << std::endl;

        // eq 12 dopo averla messa nel logaritmo

        long double eq_12_log_fraction = compute_cardinality(Z);

        Eigen::VectorXd mm = fill_m(Z);

        long double eq_12_log_product = 0;
        for (size_t k = 0; k < K; k++) {
/*
            long double test1 =log(-alpha / K);
            long double test2 = log(tgamma(mm(k) + alpha / K));
            long double test3 = log(tgamma(n - mm(k) + 1));
            long double test4 =log(tgamma(n + 1 + alpha / K));
            std::cout <<" alpha: " << alpha  << " k: " << k << " test1: " <<test1 << " test2: "<<test2 << " test3: "<<test3 << " test4: "<< test4 << endl;
*/
            eq_12_log_product += log(-alpha / K) + log(tgamma(mm(k) + alpha / K)) + log(tgamma(n - mm(k) + 1)) -
                                 log(tgamma(n + 1 + alpha / K));
        }
        long double eq_12_log = eq_12_log_fraction + eq_12_log_product; // A volte ritorna valori positivi: questo implica che P(Z)>1 che è impossibile.
        // DA RIVEDERE

        //std::cout << "eq_12_log = eq_12_log_fraction + eq_12_log_product = " << eq_12_log_fraction << " + " << eq_12_log_product << " = " << eq_12_log << std::endl;

        // log[P(X,Z)] = log[P(X|Z)P(Z)] = log[P(X|Z)] + log[P(Z)]       (log(Equation 21) + log(Equation 12))

        long double pXZ_log = eq_12_log + eq_21_log;
        //std::cout << "pXZ_log = eq_12_log + eq_21_log = " <<  eq_12_log << " + " << eq_21_log << " = " << pXZ_log << std::endl;
        std::cout << "pXZ_log = " << pXZ_log << std::endl;

        Eigen::MatrixXd Expected_A_given_XZ = (Z.transpose()*Z+pow(sigma_x/sigma_a,2)*Eigen::MatrixXd::Identity(Z.cols(), Z.cols())).inverse()*Z.transpose()*X;
        //std::cout << Expected_A_given_XZ << std::endl;
      
        //----------------------------------------------------------------------
        //FINE calcolo log[P(X|Z)] e E[A|X,Z]



        if(it>=initial_iters){
            Ret.push_back(eliminate_null_columns(Z).first);

        }

    }



    return Ret;
}



matrix_collection GibbsSampler_IBP(const double alpha,const double gamma,const double sigma_a, const double sigma_x, const double theta, const int n, MatrixXd &A, MatrixXd &X, unsigned n_iter,unsigned initial_iters){
    std::default_random_engine generator;

    /* In this algorithm matrix Z are generated keeping only non null columns,
     * this makes the algorithm to go really slow beacuse the complexity icreases.
     * WARNING: use gamma small beacuse otherwise the algorithm will be way slower
     * */


    // Initialization of Z and m
    MatrixXd Z(n,1);
    VectorXd m;

    // D:
    const unsigned D=A.cols();

    //create a vector to put the generated Z matrices:
    matrix_collection Ret;

    for (Eigen::Index it=0;it<n_iter+initial_iters;++it){

        MatrixXd Znew;

        for (Eigen::Index i=0; i<n;++i){
            Eigen::Index z_cols=Z.cols();

            MatrixXd M(z_cols,z_cols);

            M=(Z.transpose()*Z -  Eigen::MatrixXd::Identity(z_cols,z_cols)*pow(sigma_x/sigma_a,2)).inverse();

            Eigen::VectorXd z_i=Z.row(i);

            Z.row(i).setZero();

            VectorXd positions;
            auto matvec=std::make_pair(Znew,positions);
            matvec= eliminate_null_columns(Z);
            Znew=matvec.first;
            positions=matvec.second;

            Z.row(i)=z_i;
            m=fill_m(Znew);

            //update the number of observed features:
            unsigned  K= count_nonzero(m);

            Eigen::Index count=0;

            for(Eigen::Index j=0;j<K; ++j) {
                while(positions(count)==0)
                    ++count;

                double prob_zz=(m(j)-alpha)/(theta+(n-1));

                //P(X|Z)

                Z(i,count)=1;
                M= update_M(M,Z.row(i));

                long double prob_xz_1=1/(pow((Z.transpose()*Z + sigma_x*sigma_x/sigma_a/sigma_a*Eigen::MatrixXd::Identity(z_cols,z_cols)).determinant(),D*0.5) );


                MatrixXd mat=X.transpose() * (Eigen::MatrixXd::Identity(n,n) - (Z * M * Z.transpose())) * X;
                long double prob_xz_2=mat.trace()*(-1/(2*sigma_x*sigma_x));

                long double prob_xz=prob_xz_1*exp(prob_xz_2);

                Z(i,count)=0;
                M= update_M(M,Z.row(i));
                long double prob_xz_10=1/pow((Z.transpose()*Z + sigma_x*sigma_x/sigma_a/sigma_a*Eigen::MatrixXd::Identity(Z.cols(),Z.cols())).determinant(),D*0.5);

                MatrixXd mat0=X.transpose() * (Eigen::MatrixXd::Identity(n,n) - (Z * M * Z.transpose())) * X;
                long double prob_xz_20=mat.trace()*(-1/(2*sigma_x*sigma_x));

                long double prob_xz0=prob_xz_10*exp(prob_xz_20);

                long double prob_one_temp=prob_zz*prob_xz;
                long double prob_zero_temp=(1-prob_zz)*prob_xz0;
                long double prob_param=prob_one_temp/(prob_one_temp+prob_zero_temp);



                //sample from Bernoulli distribution:

                std::bernoulli_distribution distribution_bern(prob_param);

                Znew(i,j) = distribution_bern(generator) ? 1:0;
                Z(i,count)=Znew(i,j);

                ++count;
            }


            //sample the number of new features:


            unsigned UB= K>20? 10:5;//da cambiare
            Z.resize(n,K+UB);

            //update Z-part1:
            Eigen::Index j = 0;
            for (; j < Znew.cols(); ++j)
                Z.col(j) = Znew.col(j);
            for (Eigen::Index kk = Znew.cols(); kk < Z.cols(); ++kk)
                Z.col(kk).setZero();



            M = (Z.transpose() * Z -
                 Eigen::MatrixXd::Identity(Z.cols(), Z.cols()) * pow(sigma_x / sigma_a, 2)).inverse();


            double prob=tgamma(theta+alpha+n)/tgamma(theta+alpha)*tgamma(theta)/tgamma(theta+n)*gamma;

            Eigen::VectorXd prob_new(UB);
            for (unsigned itt = 0; itt < UB; ++itt) {
                double bin_prob = poissonProbability(prob, itt);
                Z(i, j+itt) = 1;
                M = update_M(M, Z.row(i));
                long double p_xz_1 = 1 / (pow((Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *
                                                                   Eigen::MatrixXd::Identity(Z.cols(),
                                                                                             Z.cols())).determinant(),
                                              D * 0.5));
                MatrixXd mat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * M * Z.transpose())) * X;
                long double p_xz_2 = mat.trace() * (-1 / (2 * sigma_x * sigma_x));

                double p_xz = p_xz_1 * exp(p_xz_2);
                prob_new(itt) = bin_prob * p_xz;
            }
            // Normalize posterior probabilities
            double sum_posterior = prob_new.sum();
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


        if(it>=initial_iters){

            Ret.push_back(eliminate_null_columns(Z).first);
        }

    }
    return Ret;
}

/*
// Function to sample A matrix
MatrixXd sample_A(const MatrixXd& Z, const MatrixXd& X, double sigma_x, double sigma_a, std::default_random_engine& generator) {
  unsigned K = Z.cols(); // Number of features
  unsigned D = X.cols(); // Dimension of data
  
  // Posterior precision and covariance
  MatrixXd Sigma_posterior_inv = (1 / (sigma_x * sigma_x)) * Z.transpose() * Z + (1 / (sigma_a * sigma_a)) * MatrixXd::Identity(K, K);
  LLT<MatrixXd> llt(Sigma_posterior_inv); // Cholesky decomposition for numerical stability
  MatrixXd Sigma_posterior = llt.solve(MatrixXd::Identity(K, K)); // Invert the precision matrix
  
  // Posterior mean
  MatrixXd mu_posterior = Sigma_posterior * (Z.transpose() * X) / (sigma_x * sigma_x);
  
  // Sample from the posterior distribution for A
  MatrixXd new_A(K, D);
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned d = 0; d < D; ++d) {
      std::normal_distribution<double> dist(mu_posterior(k, d), sqrt(Sigma_posterior(k, k)));
      new_A(k, d) = dist(generator);
    }
  }
  
  return new_A;
}*/
