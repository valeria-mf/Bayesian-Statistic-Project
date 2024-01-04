
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


    int n=50;
    double alpha=-4, theta=15;
    double sigma_x=1.5, sigma_a=1.3;
    int n_tilde=10, D=20;
    std::default_random_engine generator;
    MatrixXd Z = Eigen::MatrixXd::Zero(n, n_tilde);
    MatrixXd A(n_tilde,D) , X(n,D);
    std::bernoulli_distribution Z_initializer(0.5);
    for(unsigned i=0; i<n;++i)
        for(unsigned j=0;j<n_tilde;++j)
            Z(i, j) = Z_initializer(generator) ? 1 : 0;
    std::normal_distribution<double> A_initializer(0,sigma_a);
    for(unsigned i=0; i<n_tilde;++i)
        for(unsigned j=0;j<D;++j)
            A(i, j) = A_initializer(generator);
        for (unsigned i=0; i< n; ++i)
        {
            Eigen::RowVectorXd R=Z.row(i);
            Eigen::RowVectorXd RR=R*A;
            for(unsigned j=0; j<D;++j){
                double param=RR(j);
                std::normal_distribution<double> X_initializer(param,sigma_x);
                X(i,j)=X_initializer(generator);

        }}
    std::cout<<"MATRIX Z DA RAGGIUNGERE:\n" << Z<<"\n"<<std::endl;
    std::cout<<"QUANTI 1?:\n" << Z.sum()<<"\n"<<std::endl;



    matrix_collection Z_gen=GibbsSampler_betabernoulli(alpha,theta,sigma_x,sigma_a,n_tilde,n,A,X,5,500);

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
