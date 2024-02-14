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
    
    
    
    // Initialization of Z
    MatrixXd Z(n,1);
    std::bernoulli_distribution Z_initializer(0.5);
    for(unsigned i=0; i< n; ++i)
         Z(i, 0) = Z_initializer(generator) ? 1 : 0;
    

    // D:
    const unsigned D=A.cols();

    //create a vector to put the generated Z matrices:
    matrix_collection Ret;
    
    //create a vector to put the K values
    VectorXd K_vector(n_iter+initial_iters);
    //create a vector to put the log[P(X|Z)]
    VectorXd logPXZ_vector(n_iter+initial_iters);
    Eigen::MatrixXd Expected_A_given_XZ;
    

      for (Eigen::Index it=0;it<n_iter+initial_iters;++it){
          
        unsigned K;

        MatrixXd Znew;

        for (Eigen::Index i=0; i<n;++i){
            Eigen::Index z_cols=Z.cols();

            MatrixXd M(z_cols,z_cols);

            M=(Z.transpose()*Z +  Eigen::MatrixXd::Identity(z_cols,z_cols)*pow(sigma_x/sigma_a,2)).inverse();

            Eigen::VectorXd z_i=Z.row(i);

            Z.row(i).setZero();

            VectorXd positions;
            auto matvec=std::make_pair(Znew,positions);
            matvec= eliminate_null_columns(Z);
            Znew=matvec.first;
            positions=matvec.second;

            Z.row(i)=z_i;
            Eigen::VectorXd m=fill_m(Znew);

            //update the number of observed features:
            K= m.size();

            Eigen::Index count=0;

            for(Eigen::Index j=0;j<K; ++j) {
                while(positions(count)==0)
                    ++count;

                double prob_zz=(m(j)-alpha)/(theta+(n-1));

               
                //P(X|Z) when z_ij=1:
                Z(i, count) = 1;
                //M = update_M(M, Z.row(i));
                M=(Z.transpose()*Z +  Eigen::MatrixXd::Identity(Z.cols(),Z.cols())*pow(sigma_x/sigma_a,2)).inverse();
                //  if(i<3)
                //  std::cout << "Update di M quando z_" << i << count << " vale 1: \n" << M << std::endl;
                long double prob_xz = calculate_log_likelihood(Z,X,M,sigma_x,sigma_a,K,D,n);
                // if(i<3)
                //  std::cout << "Print di prob_xz, log_likelihood per z=1: " << prob_xz << std::endl;

                //P(X|Z) when z_ij=0:
                Z(i, count) = 0;
                //M = update_M(M, Z.row(i));
                M=(Z.transpose()*Z +  Eigen::MatrixXd::Identity(Z.cols(),Z.cols())*pow(sigma_x/sigma_a,2)).inverse();
                // if(i<3)
                // std::cout << "Update di M quando z_" << i << count << " vale 0:\n " << M << std::endl;
                long double prob_xz0 = calculate_log_likelihood(Z,X,M,sigma_x,sigma_a,K,D,n);
                //  if(i<3)
                // std::cout << "Print di prob_xz, log_likelihood per z=0: " << prob_xz0 << std::endl;
                
                Eigen::VectorXd temp_vec(2);
                temp_vec(0)=prob_xz+ log(prob_zz);
                // if(i<3)
                //  std::cout << "Print della prima log_p: " << temp_vec(0) << std::endl;
                temp_vec(1)=prob_xz0+ log(1-prob_zz);
                // if(i<3)
                //  std::cout << "Print della seconda log_p: " << temp_vec(1) << std::endl;
                long double maximum=find_max(temp_vec);
                // if(i<3)
                //  std::cout << "Print del max fra le due: " << maximum << std::endl;
                temp_vec(0)=temp_vec(0)-maximum;
                temp_vec(1)=temp_vec(1)-maximum;
                temp_vec(0)=exp(temp_vec(0));
                // if(i<3)
                // std::cout << "Prima prob: " << temp_vec(0) << std::endl;
                temp_vec(1)=exp(temp_vec(1));
                //  if(i<3)
                // std::cout << "Seconda prob: " << temp_vec(1) << std::endl;

                long double prob_param=temp_vec(0)/(temp_vec(0)+temp_vec(1));
                // if(i<3)
                //  std::cout << "prob_param, quella usata per la bernoulli: " << prob_param << std::endl;


                //sample from Bernoulli distribution:
                std::bernoulli_distribution distribution_bern(prob_param);

                Znew(i, j) = distribution_bern(generator) ? 1 : 0;
                Z(i, count) = Znew(i, j);
                // if(i<3)
                // std::cout << "Z(i,count): " << Z(i,count) << std::endl;


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



            M = (Z.transpose() * Z +
                 Eigen::MatrixXd::Identity(Z.cols(), Z.cols()) * pow(sigma_x / sigma_a, 2)).inverse();


            double prob=tgamma(theta+alpha+n)/tgamma(theta+alpha)*tgamma(theta)/tgamma(theta+n)*gamma;

            Eigen::VectorXd prob_new(UB+1);
            for (unsigned itt = 0; itt <= UB; ++itt) {
                double poi_prob = poissonProbability(prob, itt);
                if(itt>0)
                     Z(i, Znew.cols()-1 + itt) = 1;
                M=(Z.transpose()*Z +  Eigen::MatrixXd::Identity(K,K)*pow(sigma_x/sigma_a,2)).inverse();
                long double px_znewfeat= calculate_log_likelihood(Z,X,M,sigma_x,sigma_a,K+itt,D,n);
                         //if(i<3)
                            //std::cout << "px_znewfeat: " << px_znewfeat << std::endl;
                prob_new(itt) = log(poi_prob) + px_znewfeat;
                         // if(i<3)
                         //std::cout << "prob_new: " << prob_new << std::endl;
            }
             // Normalize posterior probabilities
            long double max2=find_max(prob_new);
            for (unsigned ii=0; ii<prob_new.size();++ii){
                prob_new(ii)=prob_new(ii)-max2;
                prob_new(ii)=exp(prob_new(ii));
            }
            
            double sum_posterior = prob_new.sum();
            for (unsigned l = 0; l < prob_new.size(); ++l) {
                prob_new(l) /= sum_posterior;
              // if(i<3)
              //   std::cout << "prob_new per indice = " << l << ": " << prob_new(l) << std::endl;
            }
                    
                    
            

            // Sample the number of new features based on posterior probabilities
            std::discrete_distribution<int> distribution(prob_new.data(), prob_new.data() + prob_new.size());
            int new_feat = distribution(generator);


            //update Z-part2:
           for (unsigned j=Znew.cols()+new_feat; j < Z.cols(); ++j) {
                    Z(i, j) = 0;
                }
        
            
            
        }
          
        VectorXd vect=fill_m(Z);
        K=count_nonzero(vect);

        //Alla fine di ogni iterazione calcolo la quantità log[P(X,Z)]
        //----------------------------------------------------------------------


        //Per l'IBP utilizzo Eq 14 e 26:

        

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
        long double eq_14_before_productory = compute_cardinality(Zplus) -log(tgamma(Kplus+1)) + Kplus*log(alpha)-alpha*Hn;

        Eigen::VectorXd mm = fill_m(Zplus);
        long double eq_14_log_productory = 0;
        for (size_t k = 1; k < Kplus; k++) {
            eq_14_log_productory += log(tgamma(n-mm(k)+1)) +log(tgamma(mm(k)))-log(tgamma(n+1));
        }
        long double eq_14_log = eq_14_before_productory + eq_14_log_productory; // A volte ritorna valori positivi: questo implica che P(Z)>1 che è impossibile.
        // DA RIVEDERE
        //std::cout << "eq_14_log = eq_14_before_productory + eq_14_log_productory = " << eq_14_before_productory << " + " << eq_14_log_productory << " = " << eq_14_log << std::endl;

        // log[P(X,Z)] = log[P(X|Z)P(Z)] = log[P(X|Z)] + log[P(Z)]       (log(Equation 21) + log(Equation 12))

        long double pXZ_log = eq_14_log + eq_26_log;
        //std::cout << "pXZ_log = eq_14_log + eq_26_log = " <<  eq_14_log << " + " << eq_26_log << " = " << pXZ_log << std::endl;
        //std::cout << "pXZ_log = " << pXZ_log << std::endl;

        Eigen::MatrixXd Expected_A_given_XZ = (Zplus.transpose()*Zplus+pow(sigma_x/sigma_a,2)*Eigen::MatrixXd::Identity(Zplus.cols(), Zplus.cols())).inverse()*Zplus.transpose()*X;
        //std::cout << Expected_A_given_XZ << std::endl;
        //----------------------------------------------------------------------
        //FINE calcolo log[P(X,Z)]
        
        
        
        logPXZ_vector(it)=pXZ_log;
        
        //fill the K_vector
        K_vector(it)=K;
        
        
          
        if(it>=initial_iters){

             Ret.push_back(Zplus);
        }

    }
      return Rcpp::List::create(Rcpp::Named("Z_list") = Ret, Rcpp::Named("K_vector")=K_vector,
                                Rcpp::Named("logPXZ_vector")=logPXZ_vector, Rcpp::Named("Expected_A") = Expected_A_given_XZ);;
}
