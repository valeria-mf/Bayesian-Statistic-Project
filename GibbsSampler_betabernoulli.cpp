//
// Created by Valeria Montefusco on 30/11/23.
//
#include <RcppEigen.h>
#include <Rcpp.h>

#include "auxiliar_functions.h"


//[[Rcpp::export]]
Rcpp::List GibbsSampler_betabernoulli( double alpha, double theta, double sigma_x,double sigma_a,  int n_tilde,  int n,  SEXP A_, SEXP X_, unsigned n_iter, unsigned initial_iters){

    /*STRATEGY:
     * When generating a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices  matrices Z with only non null columns will be inserted.
  */
   // Rcpp::Rcout << "Dimensioni di A: " << std::endl;
    Rcpp::NumericMatrix mat_A(A_);
    Rcpp::NumericMatrix mat_X(X_);
    
   // Rcpp::Rcout << "Numero di righe: " << mat_X.nrow() << std::endl;
   // Rcpp::Rcout << "Numero di colonne: " << mat_X.ncol() << std::endl;
    

    Eigen::Map<Eigen::MatrixXd> A(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_A));
    Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_X));
    
 //   Rcpp::Rcout << "Dimensioni di A: " << A.rows() << " x " << A.cols() << std::endl;
  //  Rcpp::Rcout << "Dimensioni di X: " << X.rows() << " x " << X.cols() << std::endl;
    
    
  
    // Initialization of Z and m:
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);
    VectorXd m(n_tilde);

    // D:
    const unsigned D=A.cols();

    //create a set to put the generated Z matrices:
    matrix_collection Ret;


    std::default_random_engine generator;


    for (Eigen::Index it=0;it<n_iter+initial_iters;++it){

        MatrixXd Znew;

        //INITIALIZE M MATRIX:
        MatrixXd M=(Z.transpose()*Z -  Eigen::MatrixXd::Identity(n_tilde,n_tilde)*pow(sigma_x/sigma_a,2)).inverse();


        for (Eigen::Index i=0; i<n;++i){

            Eigen::VectorXd z_i=Z.row(i);
          

            Z.row(i).setZero();

            VectorXd positions;
            auto matvec=std::make_pair(Znew,positions);
            matvec= eliminate_null_columns(Z);
            Znew=matvec.first; //the new matrix that I will update
            positions=matvec.second; //to see the positions where I remove the columns

            Z.row(i)=z_i;
            m=fill_m(Znew);

            //update the number of observed features:
            unsigned  K= count_nonzero(m);

            Eigen::Index count=0;

            for(Eigen::Index j=0;j<K; ++j) {
                while(positions(count)==0)
                    ++count;

                double prob_zz=(m(j)-alpha)/(theta+(n-1));

                //P(X|Z) when z_ij=1:
                Z(i,count)=1;
                M= update_M(M,Z.row(i));

                long double prob_xz_1=1/(pow((Z.transpose()*Z + sigma_x*sigma_x/sigma_a/sigma_a*Eigen::MatrixXd::Identity(n_tilde,n_tilde)).determinant(),D*0.5) );
                MatrixXd mat=X.transpose() * (Eigen::MatrixXd::Identity(n,n) - (Z * M * Z.transpose())) * X;
                long double prob_xz_2=mat.trace()*(-1/(2*sigma_x*sigma_x));
               

                long double prob_xz=prob_xz_1*exp(prob_xz_2);
                    

                //P(X|Z) when z_ij=1:
                Z(i,count)=0;
                M= update_M(M,Z.row(i));
                long double prob_xz_10=1/pow((Z.transpose()*Z + sigma_x*sigma_x/sigma_a/sigma_a*Eigen::MatrixXd::Identity(Z.cols(),Z.cols())).determinant(),D*0.5);

                MatrixXd mat0=X.transpose() * (Eigen::MatrixXd::Identity(n,n) - (Z * M * Z.transpose())) * X;
                long double prob_xz_20=mat.trace()*(-1/(2*sigma_x*sigma_x));

                long double prob_xz0=prob_xz_10*exp(prob_xz_20);

                //Bernoulli parameter:

                long double prob_one_temp=prob_zz*prob_xz;
                long double prob_zero_temp=(1-prob_zz)*prob_xz0;
                long double prob_param=prob_one_temp/(prob_one_temp+prob_zero_temp); //PROBLEM: always too small

                //sample from Bernoulli distribution:
                std::bernoulli_distribution distribution_bern(prob_param);
              
                Znew(i,j) = distribution_bern(generator) ? 1:0;
                Z(i,count)=Znew(i,j);


                ++count;
            }

            //sample the number of new features:


            double prob =1-(theta+alpha+n-1)/(theta+n-1);
            std::binomial_distribution<int> distribution_bin(n_tilde-K,prob);
            int temp=distribution_bin(generator);

            //update Z:
            Eigen::Index j=0;
            for(;j<Znew.cols();++j)
                Z.col(j)=Znew.col(j);
            for(;j<Znew.cols()+temp;++j)
            {
                Z.col(j).setZero();
                Z(i,j)+=1;
            }
            for(;j<Z.cols();++j)
                Z.col(j).setZero();

        }

        if(it>=initial_iters){
            Ret.push_back(eliminate_null_columns(Z).first);

        }

    }



    return Rcpp::List::create(Rcpp::Named("result") = Ret);
}
