#include <iostream>

#include "auxiliar_functions.h"
#include "GibbsSampler_betabernoulli.h"
#include <list>
#include <random>
#include <set>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/LU>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::cout;
using std::endl;

int main() {




    int seed = 1234;

    int n = 10;  // # objects
    int K = 10;  // # displayed features
    int D = 20;  // # observable properties (each represented by a row vector x_i
    int sigma_x = 2;
    int sigma_a = 3;
    int mean_a = 1;

    int alpha = -7;
    int theta = 16;
    int n_tilde = 30;
    unsigned n_iter = 20;
    unsigned initial_iters = 25;




    MatrixXd Z_true(n, K);
    MatrixXd A(K, D);
    MatrixXd X = MatrixXd::Zero(n, D);

    std::seed_seq seedSequence{seed};
    std::default_random_engine rng(seedSequence);

    std::uniform_int_distribution<int> uniform_dist(0, 1);
    std::normal_distribution<double> normal_dist_a(mean_a, sigma_a);



    // Fill the matrix Z_true with random 0s and 1s
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < K; ++j) {
            Z_true(i, j) = uniform_dist(rng);
        }
    }

    // Fill A normally distributed
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < D; ++j) {
            A(i, j) = normal_dist_a(rng);
        }
    }

    // Fill X (Cholesky Decomposition) (X represents the D properties of all n objects)
    for (int i = 0; i < n; ++i) {
        VectorXd mu = Z_true.row(i) * A;
        MatrixXd Id = MatrixXd::Identity(D, D);
        MatrixXd Sigma = sigma_x * sigma_x * Id;

        // Generate multivariate normal sample
        Eigen::LLT<MatrixXd> llt(Sigma); // Cholesky decomposition for efficiency
        VectorXd sample = mu + llt.matrixL() * VectorXd::Random(D);

        X.row(i) = sample;
    }



    std::list<MatrixXd> final = GibbsSampler_betabernoulli(alpha, theta, sigma_x, sigma_a, n_tilde, n, A, X, n_iter, initial_iters);


    /*
    for (const auto& matrix : final) {
        std::cout << matrix << std::endl;
        std::cout << "-----" << std::endl;  // Separating line for clarity
    }
    */


    return 42;
}