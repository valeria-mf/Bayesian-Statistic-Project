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
Rcpp::List GibbsSampler_betabernoulli( double alpha, double theta, double sigma_x, double a_x, double b_x, double a_a, double b_a, 
                                       double a, double b, double c,  int n, SEXP X_, unsigned n_iter, unsigned initial_iters, unsigned UB){

    /*STRATEGY:
     * When generating a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices  matrices Z with only non null columns will be inserted.
  */
   // Rcpp::Rcout << "Dimensioni di A: " << std::endl;
    
    Rcpp::NumericMatrix mat_X(X_);

    Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_X));
    
    std::default_random_engine generator;

  
// D:
    const unsigned D = X.cols();
 
    // Initialization of Z and m:
    MatrixXd Z = Eigen::MatrixXd::Zero(n, 1);
    VectorXd m;
    
   // X : n x D
   // Z : n x K
   // A : K x D
   // X = Z * A

    std::bernoulli_distribution Z_initializer(0.5);
    for(unsigned i=0; i< n ; ++i)
            Z(i, 0) = Z_initializer(generator) ? 1 : 0;
  //  std::cout << Z << std::endl;

 //Initialization of A:
  int n_tilde = 1;
  MatrixXd A = Eigen::MatrixXd::Zero(n_tilde, D);
  std::normal_distribution<double> A_initializer(0,1);
  for(unsigned j=0;j<D;++j)
      A(0, j) = A_initializer(generator);
  

  // prior of A
    std::gamma_distribution<double> distr(a, b);  
    double prior_precision_a = pow(distr(generator),-1); // sampling from a gamma
    double sigma_a = 1/prior_precision_a;
    

    //create a set to put the generated Z matrices:
    matrix_collection Ret;
    
    //create a vector to put the K values
    VectorXd K_vector(n_iter+initial_iters);
    //create a vector to put the log[P(X|Z)]
    VectorXd logPXZ_vector(n_iter+initial_iters);
    VectorXd sigmaA_vector(n_iter+initial_iters);
    VectorXd sigmaX_vector(n_iter+initial_iters);
    Eigen::MatrixXd Expected_A_given_XZ;

    

  // initialization of parameters (for the sample2_A)
    double mu_mean = 0;
    double mu_var = c*sigma_a^2;
    
  
    for (Eigen::Index it=0;it<n_iter+initial_iters;++it){

        n_tilde = A.rows();
        MatrixXd Znew;


        for (Eigen::Index i=0; i<n;++i) {

            Eigen::Index z_cols=Z.cols();
            MatrixXd M(z_cols,z_cols);


            Eigen::VectorXd z_i = Z.row(i);

            Z.row(i).setZero();

            VectorXd positions;
            auto matvec = std::make_pair(Znew, positions);
            matvec = eliminate_null_columns(Z);
            Znew = matvec.first; //the new matrix that I will update
            positions = matvec.second; //to see the positions where I remove the columns

            M=(Znew.transpose()*Znew -  Eigen::MatrixXd::Identity(z_cols,z_cols)*pow(sigma_x/sigma_a,2)).inverse();


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



                //P(X|Z) when z_ij=0:
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

            Znew = Z; // Znew senza colonne nulle
            Z.resize(n,K+UB); // Z ora avrà UB nuove colonne nulle

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
                double poi_prob = poissonProbability(prob, itt);
                Z(i, j + itt) = 1;
                M = update_M(M, Z.row(i));

                long double p_xz = calculate_likelihood(Z,X,M,sigma_x,sigma_a,n_tilde,D,n);
                prob_new(itt) = poi_prob * p_xz;
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
        double proposal_variance_factor_sigma_x = 0.1 * sigma_x; // e.g., 10% of current sigma_x
        double proposal_variance_factor_sigma_a = 0.1 * sigma_a; // e.g., 10% of current sigma_a
        
        
        sigma_x = metropolis_step_sigma_x(sigma_x,Z,X,A,sigma_a,proposal_variance_factor_sigma_x,generator,a_x, b_x);
        //aggiungo anche commento per sigma_a che non c'era, forse qui non serve?
        //sigma_a = metropolis_step_sigma_a(sigma_a,Z,X,A,sigma_x,proposal_variance_factor_sigma_a,generator,a_a, b_a);

        
      
        A = sample2_A(Z, X, A, &a, &b, &mu_mean, &mu_var, generator); // update of A




        //Alla fine di ogni iterazione calcolo la quantità log[P(X,Z)]
        //----------------------------------------------------------------------


        //Per l'IBP utilizzo Eq 14 e 26:

        int K = A.rows();
        int D = A.cols();

        Eigen::MatrixXd Zplus = eliminate_null_columns(Z).first;
        int Kplus = Zplus.cols();

        // Eq 26 dopo averla messa nel logaritmo:

        long double eq_26_log_denominator = -(n * D / 2) * log(2 * 3.14159265359) - (n - Kplus) * D * log(sigma_x) - Kplus * D * log(sigma_a) - D / 2 * log((Zplus.transpose() * Zplus + sigma_x *sigma_x /sigma_a /sigma_a * Eigen::MatrixXd::Identity(Zplus.cols(),Zplus.cols())).determinant());

        Eigen::MatrixXd MM = (Zplus.transpose() * Zplus + sigma_x * sigma_x / sigma_a / sigma_a *
                                                          Eigen::MatrixXd::Identity(Zplus.cols(), Zplus.cols())).inverse();

        MatrixXd matmat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Zplus * MM * Zplus.transpose())) * X;
        long double eq_26_log_exponential = -matmat.trace() / sigma_x / sigma_x / 2;
        long double eq_26_log = eq_26_log_denominator + eq_26_log_exponential;
        //std::cout << "eq_26_log = eq_26_log_denominator + eq_26_log_exponential = " << eq_26_log_denominator << " + " << eq_26_log_exponential << " = " << eq_26_log << std::endl;


        // eq 14 dopo averla messa nel logaritmo

        long double Hn = 0;
        for(int cont = 1; cont <= n; cont++)
        {Hn += 1/cont;}
        long double eq_14_before_productory = compute_cardinality(Zplus) -log(tgamma(Kplus)) + Kplus*log(alpha)-alpha*Hn;

        Eigen::VectorXd mm = fill_m(Zplus);
        long double eq_14_log_productory = 0;
        for (size_t k = 1; k < Kplus; k++) {
            eq_14_log_productory += log(tgamma(n-mm(k))) +log(tgamma(mm(k)-1))-log(tgamma(n));
        }
        long double eq_14_log = eq_14_before_productory + eq_14_log_productory; // A volte ritorna valori positivi: questo implica che P(Z)>1 che è impossibile.
        // DA RIVEDERE
        //std::cout << "eq_14_log = eq_14_before_productory + eq_14_log_productory = " << eq_14_before_productory << " + " << eq_14_log_productory << " = " << eq_14_log << std::endl;

        // log[P(X,Z)] = log[P(X|Z)P(Z)] = log[P(X|Z)] + log[P(Z)]       (log(Equation 21) + log(Equation 12))

        long double pXZ_log = eq_14_log + eq_26_log;
        //std::cout << "pXZ_log = eq_14_log + eq_26_log = " <<  eq_14_log << " + " << eq_26_log << " = " << pXZ_log << std::endl;
        //std::cout << "pXZ_log = " << pXZ_log << std::endl;

        Eigen::MatrixXd Expected_A_given_XZ = (Zplus.transpose()*Zplus+pow(sigma_x/sigma_a,2)*Eigen::MatrixXd::Identity(Zplus.cols(), Zplus.cols())).inverse()*Zplus.transpose()*X;
        std::cout << Expected_A_given_XZ << std::endl;
        //----------------------------------------------------------------------
        //FINE calcolo log[P(X,Z)]
        
        logPXZ_vector(it)=pXZ_log;
      
        //fill the K_vector
        VectorXd vect=fill_m(Z);
        K_vector(it)=count_nonzero(vect);
      
        sigmaX_vector(it)=sigma_x;
        sigmaA_vector(it)=sigma_a;




        
        if(it>=initial_iters)
              Ret.push_back(eliminate_null_columns(Z).first);
        }
    return Rcpp::List::create(Rcpp::Named("Z_list") = Ret, Rcpp::Named("K_vector")=K_vector, Rcpp::Named("logPXZ_vector")=logPXZ_vector, Rcpp::Named("Expected_A") = Expected_A_given_XZ,
                                          Rcpp::Named("sigmaA_vector")=sigmaA_vector, Rcpp::Named("sigmaX_vector")=sigmaX_vector);
}
