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
Rcpp::List GibbsSampler_betabernoulli( double alpha, double theta, double sigma_x,double sigma_a,  int n_tilde,  int n,  SEXP A_, SEXP X_, unsigned n_iter, unsigned initial_iters){

    /*STRATEGY:
     * When generating a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices  matrices Z with only non null columns will be inserted.
  */
   // Rcpp::Rcout << "Dimensioni di A: " << std::endl;
    Rcpp::NumericMatrix mat_A(A_);
    Rcpp::NumericMatrix mat_X(X_);

    Eigen::Map<Eigen::MatrixXd> A(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_A));
    Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_X));

  
    // Initialization of Z
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);
   
    std::default_random_engine generator;

    std::bernoulli_distribution Z_initializer(0.5);
    for(unsigned i=0; i< n ; ++i)
        //for(unsigned j=0; j<n_tilde;++j)
            Z(i, 0) = Z_initializer(generator) ? 1 : 0;
  //  std::cout << Z << std::endl;



    // D:
    const unsigned D = A.cols();

    //create a set to put the generated Z matrices:
    matrix_collection Ret;
    
    //create a vector to put the K values
    VectorXd K_vector(n_iter+initial_iters);
    //create a vector to put the log[P(X|Z)]
    VectorXd logPXZ_vector(n_iter+initial_iters);
    Eigen::MatrixXd Expected_A_given_XZ;
    



    for (Eigen::Index it=0;it<n_iter+initial_iters;++it){

        unsigned K;

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
            Eigen:VectorXd m = fill_m(Znew);

            //update the number of observed features:
            K = m.size();
            std::cout << K << std::endl;
            

            Eigen::Index count = 0;

            for (Eigen::Index j = 0; j < K; ++j) {
                while (positions(count) == 0)
                    ++count;

                double prob_zz = (m(j) - alpha) / (theta + (n - 1));

                //P(X|Z) when z_ij=1:
                Z(i, count) = 1;
                M = update_M(M, Z.row(i));

              
                long double prob_xz = calculate_log_likelihood(Z,X,M,sigma_x,sigma_a,K,D,n);

                //P(X|Z) when z_ij=0:
                Z(i, count) = 0;
                M = update_M(M, Z.row(i));
                long double prob_xz0 = calculate_log_likelihood(Z,X,M,sigma_x,sigma_a,n_tilde,D,n);
                Eigen::VectorXd temp_vec(2);
                temp_vec(0)=prob_xz+ log(prob_zz);
                temp_vec(1)=prob_xz0+ log(1-prob_zz);
                long double maximum=find_max(temp_vec);
                temp_vec(0)=temp_vec(0)-maximum;
                temp_vec(1)=temp_vec(1)-maximum;
                temp_vec(0)=exp(temp_vec(0));
                temp_vec(1)=exp(temp_vec(1));

                long double prob_param=temp_vec(0)/(temp_vec(0)+temp_vec(1));
                
                std::cout <<"Prob_param:"<< prob_param << std::endl;


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
                        long double bin_prob = binomialProbability(n_res, prob, itt);
    
                        Z(i, Znew.cols() + itt) = 1;
                        M = update_M(M, Z.row(i));
                        long double px_znewfeat= calculate_log_likelihood(Z,X,M,sigma_x,sigma_a,K+itt,D,n);
                        prob_new(itt) = log(bin_prob) + px_znewfeat;
    
                    }
                    // Normalize posterior probabilities
                    long double max2=find_max(prob_new);
                    for (unsigned ii=0; ii<prob_new.size();++ii){
                        prob_new(ii)=prob_new(ii)-max2;
                        prob_new(ii)=exp(prob_new(ii));
                    }
                    std::cout << "Prob_new:" << std::endl;
                    double sum_posterior = prob_new.sum();
                    for (unsigned l = 0; l < prob_new.size(); ++l) {
                        prob_new(l) /= sum_posterior;
                        std::cout << prob_new(l) << std::endl;
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
        VectorXd vect=fill_m(Z);
        K=count_nonzero(vect);



        
        //Alla fine di ogni iterazione calcolo la quantità log[P(X|Z)]
        //----------------------------------------------------------------------

        //Per il BB utilizzo Eq 21 e 12:

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




        
        if(it>=initial_iters)
              Ret.push_back(eliminate_null_columns(Z).first);
        }
    return Rcpp::List::create(Rcpp::Named("Z_list") = Ret, Rcpp::Named("K_vector")=K_vector, Rcpp::Named("logPXZ_vector")=logPXZ_vector, Rcpp::Named("Expected_A") = Expected_A_given_XZ);
}
