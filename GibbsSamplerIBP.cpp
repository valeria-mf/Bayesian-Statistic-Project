//
// Created by Valeria Montefusco on 30/11/23.
//
#include <RcppEigen.h>
#include <Rcpp.h>

#include "auxiliar_functions.h"



//[[Rcpp::export]]
Rcpp::List GibbsSampler_IBP(const double alpha,const double gamma,const double sigma_a, const double sigma_x, const double theta, const int n, SEXP A_, SEXP X_, unsigned UB, unsigned n_iter,unsigned initial_iters){
    std::default_random_engine generator;

    /* In this algorithm matrix Z are generated keeping only non null columns,
     * this makes the algorithm to go really slow beacuse the complexity icreases.
     * WARNING: use gamma small beacuse otherwise the algorithm will be way slower
     * */

    Rcpp::NumericMatrix mat_A(A_);
    Rcpp::NumericMatrix mat_X(X_);
    
 
    Eigen::Map<Eigen::MatrixXd> A(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_A));
    Eigen::Map<Eigen::MatrixXd> X(Rcpp::as<Eigen::Map<Eigen::MatrixXd>>(mat_X));
    

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
                double poi_prob = poissonProbability(prob, itt);
                Z(i, j+itt) = 1;
                M = update_M(M, Z.row(i));
                long double p_xz_1 = 1 / (pow((Z.transpose() * Z + sigma_x * sigma_x / sigma_a / sigma_a *
                                                                   Eigen::MatrixXd::Identity(Z.cols(),
                                                                                             Z.cols())).determinant(),
                                              D * 0.5));
                MatrixXd mat = X.transpose() * (Eigen::MatrixXd::Identity(n, n) - (Z * M * Z.transpose())) * X;
                long double p_xz_2 = mat.trace() * (-1 / (2 * sigma_x * sigma_x));

                double p_xz = p_xz_1 * exp(p_xz_2);
                prob_new(itt) = poi_prob * p_xz;
            }
            // Normalize posterior probabilities
            double sum_posterior = prob_new.sum();
            for (unsigned l=0;l<prob_new.size();++l) {
                prob_new(l) /= sum_posterior;
            }

            // Sample the number of new features based on posterior probabilities
            std::discrete_distribution<int> distribution(prob_new.data(), prob_new.data() + prob_new.size());
            int new_feat = distribution(generator);


            //update Z-part2:
            j--;
            for (; j >= Znew.cols() + new_feat; --j) {
                Z(i, j) = 0;
            }
            
            
            
        }

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
          
        if(it>=initial_iters){

             Ret.push_back(Z);
        }

    }
      return Rcpp::List::create(Rcpp::Named("Z_list") = Ret, Rcpp::Named("K_vector")=K_vector,
                                Rcpp::Named("logPXZ_vector")=logPXZ_vector, Rcpp::Named("Expected_A") = Expected_A_given_XZ);;
}
