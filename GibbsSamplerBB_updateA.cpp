//
// Created by Valeria Montefusco on 30/11/23.
//
#include <RcppEigen.h>
#include <Rcpp.h>

#include "auxiliar_functions.h"

#include <cmath>
#include <random>
#define pi 3.1415926535897932



//[[Rcpp::export]]
Rcpp::List GibbsSampler_betabernoulli( double alpha, double theta, double sigma_x,double sigma_a, double a_x, double b_x, 
                                       double a_a, double b_a, int n_tilde,  int n, SEXP X_, unsigned n_iter, unsigned initial_iters){

    /*STRATEGY:
     * When generating a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices  matrices Z with only non null columns will be inserted.
  */
   // Rcpp::Rcout << "Dimensioni di A: " << std::endl;
   // Rcpp::NumericMatrix mat_A(A_);
    Rcpp::NumericMatrix mat_X(X_);

   // Eigen::Map<Eigen::MatrixXd> A(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_A));
    Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_X));

  std::default_random_engine generator;

  // D:
    const unsigned D = X.cols();
 
    // Initialization of Z and m:
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);
    VectorXd m(n_tilde);
    
   

    std::bernoulli_distribution Z_initializer(0.5);
    for(unsigned i=0; i< n ; ++i)
      for(unsigned j=0; j<n_tilde; ++j)
            Z(i, j) = Z_initializer(generator) ? 1 : 0;
  //  std::cout << Z << std::endl;

 //Initialization of A:
  MatrixXd A = Eigen::MatrixXd::Zero(n_tilde, D);
  std::normal_distribution<double> A_initializer(0,1);
  for(unsigned i=0; i<n_tilde;++i)
        for(unsigned j=0;j<D;++j)
            A(i, j) = A_initializer(generator);
  

    

    //create a set to put the generated Z matrices:
    matrix_collection Ret;
    
    //create a vector to put the K values
    VectorXd K_vector(n_iter+initial_iters);
    //create a vector to put the log[P(X|Z)]
    VectorXd logPXZ_vector(n_iter+initial_iters);
    VectorXd sigmaA_vector(n_iter+initial_iters);
    VectorXd sigmaX_vector(n_iter+initial_iters);
    Eigen::MatrixXd Expected_A_given_XZ;
    



    for (Eigen::Index it=0;it<n_iter+initial_iters;++it){

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

                long double prob_xz= calculate_likelihood(Z,X,M,sigma_x,sigma_a,n_tilde,D,n);



                //P(X|Z) when z_ij=1:
                Z(i, count) = 0;
                M = update_M(M, Z.row(i));
                
                long double prob_xz0 = calculate_likelihood(Z,X,M,sigma_x,sigma_a,n_tilde,D,n);


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
                        
                        long double p_xz = calculate_likelihood(Z,X,M,sigma_x,sigma_a,n_tilde,D,n);
                        prob_new(itt) = bin_prob * p_xz;
                    }
                    // Normalize posterior probabilities
                    double sum_posterior = prob_new.sum();
                    for (unsigned l = 0; l < prob_new.size(); ++l) {
                        prob_new(l) /= sum_posterior;
                    }

                    // Sample the number of new features based on posterior probabilities
                    //std::discrete_distribution<int> distribution(prob_new.begin(), prob_new.end());
                    std::discrete_distribution<int> distribution(prob_new.data(), prob_new.data() + prob_new.size());
                    int new_feat = distribution(generator);


                    //update Z-part2:
                    j--;
                    for (; j >= Znew.cols() + new_feat; --j) {
                        Z(i, j) = 0;
                    }
                }
        }
        double proposal_variance_factor_sigma_x = 0.1 * sigma_x; // e.g., 10% of current sigma_x
        double proposal_variance_factor_sigma_a = 0.1 * sigma_a; // e.g., 10% of current sigma_a
        
        
        sigma_x = metropolis_step_sigma_x(sigma_x,Z,X,A,sigma_a,proposal_variance_factor_sigma_x,generator,a_x, b_x);
        sigma_a = metropolis_step_sigma_a(sigma_a,Z,X,A,sigma_x,proposal_variance_factor_sigma_a,generator,a_a, b_a);
        
        
        A= sample_A(Z, X, sigma_x, sigma_a, generator);



        
        //Alla fine di ogni iterazione calcolo la quantità log[P(X|Z)]
        //----------------------------------------------------------------------

        //Per il BB utilizzo Eq 21 e 12:

        int K = A.rows();
        int D = A.cols();

        // Eq 21 dopo averla messa nel logaritmo:

        long double eq_21_log_denominator = -(n * D / 2) * log(2 * pi) - (n - K) * D * log(sigma_x) - K * D * log(sigma_a) - D / 2 *log((Z.transpose() *Z + sigma_x * sigma_x /sigma_a /sigma_a *Eigen::MatrixXd::Identity(Z.cols(), Z.cols())).determinant());                                                                                    //

        Eigen::MatrixXd MM = (Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *Eigen::MatrixXd::Identity(Z.cols(), Z.cols())).inverse();

        MatrixXd matmat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * MM * Z.transpose())) * X;
        long double eq_21_log_exponential = -matmat.trace() / sigma_x / sigma_x / 2;
        long double eq_21_log = eq_21_log_denominator + eq_21_log_exponential;
        
        // Rcpp::Rcout << "eq_21_log_denominator + eq_21_log_exponential = " << eq_21_log_denominator << " + " << eq_21_log_exponential << " = " << eq_21_log << endl;

        // eq 12 dopo averla messa nel logaritmo

        long double eq_12_log_fraction = compute_cardinality(Z);

        Eigen::VectorXd mm = fill_m(Z);
        long double eq_12_log_product = 0;
        for (size_t k = 0; k < K; k++) {
            eq_12_log_product += log(-alpha / K) + log(tgamma(mm(k) + alpha / K)) + log(tgamma(n - mm(k) + 1)) - log(tgamma(n + 1 + alpha / K));
        }
        long double eq_12_log = eq_12_log_fraction + eq_12_log_product; // A volte ritorna valori positivi: questo implica che P(Z)>1 che è impossibile.
                                                                        // CALCOLI DA RIVEDERE

        // Rcpp::Rcout << "eq_12_log = eq_12_log_fraction + eq_12_log_product = " << eq_12_log_fraction << " + " << eq_12_log_product << " = " << eq_12_log << endl;

        // log[P(X,Z)] = log[P(X|Z)P(Z)] = log[P(X|Z)] + log[P(Z)]       (log(Equation 21) + log(Equation 12))

        long double pXZ_log = eq_12_log + eq_21_log;
        // Rcpp::Rcout << "pXZ_log = eq_12_log + eq_21_log = " <<  eq_12_log << " + " << eq_21_log << " = " << pXZ_log << std::endl;
        
        Expected_A_given_XZ = (Z.transpose()*Z+pow(sigma_x/sigma_a,2)*Eigen::MatrixXd::Identity(Z.cols(), Z.cols())).inverse()*Z.transpose()*X;
        
        //----------------------------------------------------------------------
        //FINE calcolo log[P(X|Z)]
        
        logPXZ_vector(it)=pXZ_log;
        //fill the K_vector
        K_vector(it)=K;
        sigmaX_vector(it)=sigma_x;
        sigmaA_vector(it)=sigma_a;




        
        if(it>=initial_iters)
              Ret.push_back(eliminate_null_columns(Z).first);
        }
    return Rcpp::List::create(Rcpp::Named("Z_list") = Ret, Rcpp::Named("K_vector")=K_vector, Rcpp::Named("logPXZ_vector")=logPXZ_vector, Rcpp::Named("Expected_A") = Expected_A_given_XZ,
                                          Rcpp::Named("sigmaA_vector")=sigmaA_vector, Rcpp::Named("sigmaX_vector")=sigmaX_vector);
}
