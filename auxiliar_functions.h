//
// Created by Valeria Montefusco on 30/11/23.
//

//#ifndef PROGETTO_CON_LIBRERIE_AUXILIAR_FUNCTIONS_H
//#define PROGETTO_CON_LIBRERIE_AUXILIAR_FUNCTIONS_H


//#endif //PROGETTO_CON_LIBRERIE_AUXILIAR_FUNCTIONS_H
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/LU"
#include "opencl-c-base.h"
#include "random"
#include <vector>

using Eigen::VectorXd;

using Eigen::MatrixXd;
using Eigen::VectorXd;


typedef std::vector<Eigen::MatrixXd> matrix_collection;




// matrix_collection GibbsSampler_IBP(const double alpha,const double gamma, const double theta, const double sigma_a, const double sigma_x, const int n, MatrixXd &A,MatrixXd &X, unsigned n_iter=1000, unsigned initial_iters=10);

Eigen::VectorXd fill_m(MatrixXd Z);
unsigned count_nonzero(Eigen::VectorXd & m);
std::pair<Eigen::MatrixXd, Eigen::VectorXd>eliminate_null_columns(MatrixXd &Z);
Eigen::MatrixXd update_M(MatrixXd &M,VectorXd z_i);
unsigned find_max(Eigen::VectorXd &vec);
unsigned factorial(unsigned n);
double binomialProbability(unsigned n_res, double prob, unsigned k);
double poissonProbability(int k, double lambda);

Eigen::VectorXd fill_m(MatrixXd Z){
    Eigen::VectorXd m(Z.cols());
    m.setZero();
    for(size_t i=0; i< Z.cols();++i)
        m(i)=Z.col(i).sum();
    return m;
}
unsigned count_nonzero(Eigen::VectorXd &m){
    unsigned count=0;
    for (size_t j=0; j<m.size();++j)
        if (m(j)>0)
            ++count;
    return count;
}


std::pair<Eigen::MatrixXd, Eigen::VectorXd> eliminate_null_columns(Eigen::MatrixXd &Z) {
    size_t cols = Z.cols();
    Eigen::VectorXd m=fill_m(Z);
    size_t colnew=count_nonzero(m);
    MatrixXd Ret(Z.rows(), colnew);
    Ret.setZero();
    VectorXd ret(cols);
    ret.setZero();
    size_t iter=0;

    for (size_t i = 0; i < cols &&iter<colnew; ++i) {
        if (Z.col(i).norm() > 0) {
            Ret.col(iter)=Z.col(i);
            ret(i)=1;
            ++iter;
        }

    }

    return std::make_pair(Ret,ret);
}



Eigen::MatrixXd update_M(MatrixXd &M,VectorXd z_i){

    MatrixXd M_i= M - (M*z_i*z_i.transpose()*M)/((z_i.transpose()*M*z_i) - 1);
    M=M_i - (M_i*z_i*z_i.transpose()*M_i)/((z_i.transpose()*M_i*z_i) +1 );
    return M;

}


#include <iostream>
#include <cmath>

#include <iostream>
#include <cmath>

// Function to calculate factorial (used for binomial coefficient)
unsigned factorial(unsigned n) {
    if (n == 0 || n == 1)
        return 1;
    else
        return n * factorial(n - 1);
}

// Function to calculate the probability of k = certain value in binomial distribution
double binomialProbability(unsigned n_res, double prob, unsigned k) {
    double binomial_coefficient = factorial(n_res) / (factorial(k) * factorial(n_res - k));
    return binomial_coefficient * pow(prob, k) * pow(1 - prob, n_res - k);
}

double poissonProbability(int k, double lambda) {
    return (exp(-lambda) * pow(lambda, k)) / tgamma(k + 1);
}


unsigned find_max(Eigen::VectorXd &vec){
    unsigned ret=0;
    double val=vec(0);
    for(int i=1; i<vec.size();++i){
        if (vec(i)>val){
            val=vec(i);
            ret=i;
        }
    }
    return ret;
}
