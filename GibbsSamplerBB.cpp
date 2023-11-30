//
// Created by Valeria Montefusco on 30/11/23.
//

#include "auxiliar_functions.h"


matrix_collection GibbsSampler_betabernoulli( double alpha, double theta, double sigma_x,double sigma_a,  int n_tilde,  int n,  MatrixXd &A, MatrixXd &X, unsigned n_iter, unsigned initial_iters){

    /*STRATEGY:
     * When generatig a new matrix the null columns will be moved at the end instead of being removed.
     * * Anyway, in the vector of matrices will be inserted matrices Z with only non null columns.
  */


    // Initialization of Z and m:
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);
    VectorXd m(n_tilde);

    // D:
    const unsigned D=A.cols();

    //create a set to put the generated Z matrices:
    matrix_collection Ret;


    std::default_random_engine generator;


    for (size_t it=0;it<n_iter+initial_iters;++it){

        MatrixXd Znew;

        //INITIALIZE M MATRIX:
        MatrixXd M=(Z.transpose()*Z -  Eigen::MatrixXd::Identity(n_tilde,n_tilde)*pow(sigma_x/sigma_a,2)).inverse();


        for (size_t i=0; i<n;++i){

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

            size_t count=0;

            for(size_t j=0;j<K; ++j) {
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
            size_t j=0;
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



    return Ret;
}
