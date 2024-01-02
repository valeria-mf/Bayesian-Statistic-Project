
#include "auxiliar_functions.h"
#include "truncated_normal.h"
/*OSSERVAZIONI:
 * alpha e theta regolano il numero di features che vengono di nuovo osservate e quelle invece nuove
 * nell'IBP, maggiore è gamma, maggiore è il nuovo numero di  nuove features osservate, per questo l'algoritmo diventa più lento
 *
 */


//matrix_collection gibbsSamplerBetabernoulli(double alpha, double theta, double x, double a, int tilde, int n, MatrixXd matrix, MatrixXd matrix1, int i, int i1);

int main() {

    //parametrs in th finite case:

    int n=15;
    int D=30;
    double alpha=-7, theta=13;
    double sigma_x=2, sigma_a=5;
    int n_tilde=20;
    int n_iter = 60;
    int initial_iters = 40;
    MatrixXd A= Eigen::MatrixXd::Random(n_tilde,D);
    MatrixXd X= Eigen::MatrixXd::Random(n,D);


    matrix_collection Z_gen= gibbsSamplerBetabernoulli(alpha, theta, sigma_x, sigma_a, n_tilde, n, A, X, n_iter, initial_iters);


/* // Per inizializzare meglio il BB (Ma c'è qualcosa che non va con le dimensioni delle matrici quando faccio matrix-mumltiplications, da ricontrollare)

    int K = 10;
    int mean_a = 1;

    int seed = 1234;

    MatrixXd Z_true(n, K);
    MatrixXd A(n_tilde, D);
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
    for (int i = 0; i < n_tilde; ++i) {
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

    matrix_collection final = GibbsSampler_betabernoulli(alpha, theta, sigma_x, sigma_a, n_tilde, n, A, X, n_iter, initial_iters);
 */



/*  // Per runnare IBP
    int n=4;
    int n_tilde=15, D=12;
    MatrixXd A= Eigen::MatrixXd::Random(n_tilde,D) , X= Eigen::MatrixXd::Random(n,n_tilde);
    unsigned n_iter=10;
    //parameters IBP:
    double alpha =0.1, gamma=1.2, theta=0.3;
    double sigma_x=1.5, sigma_a=3;

    matrix_collection Z_gen=GibbsSampler_IBP( alpha, gamma,  sigma_a,  sigma_x, theta,  n,A,X,3,10);


    for(auto&iter:Z_gen)
        std::cout<<"MATRIX \n" << iter<<"\n"<<std::endl;
    */

}
