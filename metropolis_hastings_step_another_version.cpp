#include "auxiliar_functions.h"
//qualche modifica (forse non proprio esatta) al precedente algoritmo. In particolare questo serve per SIGMA_A
double metropolis_step_sigma_a(double current_sigma, const MatrixXd& Z, const MatrixXd& X,
                             const MatrixXd& A, double std_dev, std::default_random_engine& generator, double sigma,
                            double prior_variance=1) {
    //   std::normal_distribution<double> proposal_dist(current_sigma, sqrt(proposal_variance));
    //  double new_sigma = proposal_dist(generator);
    unsigned K=A.rows(); int D=A.cols();


    // if (new_sigma <= 0) return current_sigma; // Ensure proposed sigma is positive
    //double new_sigma = truncated_normal_a_sample(current_sigma,0,29);
    int seed=29;
    double new_sigma= truncated_normal_a_sample(current_sigma,std_dev,0.0,seed);

    //  double current_likelihood, new_likelihood;;
    double ratio;


        ratio =1/(pow(current_sigma/new_sigma, K*D)* exp(-1/2*((A.transpose()*A).trace()*(1/pow(current_sigma,2)-1/ pow(new_sigma,2))+current_sigma-new_sigma)));

    return ratio>1? new_sigma:current_sigma;

}
