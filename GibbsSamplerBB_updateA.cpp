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
      /*STRATEGY:
     * When generating a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices  matrices Z with only non null columns will be inserted.
  */
   // Rcpp::Rcout << "Dimensioni di A: " << std::endl;
    Rcpp::NumericMatrix mat_A(A_);
    Rcpp::NumericMatrix mat_X(X_);

    Eigen::Map<Eigen::MatrixXd> A(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_A));
    Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_X));

  
    // Initialization of Z and m:
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);
    VectorXd m(n_tilde);

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

         // Update sigma_x and sigma_a using Metropolis-Hastings steps

       //si potrebbe fare da quando it=1 aggiungendo un if
        double proposal_variance_factor_sigma_x = 0.1 * sigma_x; // e.g., 10% of current sigma_x
        double proposal_variance_factor_sigma_a = 0.1 * sigma_a; // e.g., 10% of current sigma_a
        
        sigma_x = metropolis_step_sigma(sigma_x, Z, X, A, proposal_variance_factor_sigma_x, generator, true, prior_variance_sigma_x);
        sigma_a = metropolis_step_sigma(sigma_a, Z, X, A, proposal_variance_factor_sigma_a, generator, false, prior_variance_sigma_a);
    

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


                //P(X|Z) when z_ij=1:
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

    MatrixXd A_updated = sample_A(Z, X, sigma_x, sigma_a, generator)
    

    if (it >= initial_iters) {
      Ret.push_back(eliminate_null_columns(Z).first);
    }

 return Rcpp::List::create(Rcpp::Named("result") = Ret);

}

