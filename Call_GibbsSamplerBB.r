install.packages("Rcpp")
install.packages("RcppEigen")

library(Rcpp)
library(RcppEigen)
library(MASS)

#Questo è il path del file header contenuto nella cartella unzippata dell'installazione di Eigen
Sys.setenv("PKG_CXXFLAGS"="-I\"C:\\Users\\ipval\\Desktop\\Progetto Bayesiana\\eigen-3.4.0\\eigen-3.4.0\"")

#Questo è il path del file creato con il comando install.packages("RcppEigen")
Sys.setenv(PKG_CXXFLAGS=paste0("-I", shQuote("C:/Users/ipval/AppData/Local/R/win-library/4.2/RcppEigen/include")))

Rcpp::sourceCpp("C:/Users/ipval/Desktop/Progetto Bayesiana/GibbsSamplerBB.cpp")
set.seed(1234)

N=10
K=10
D=20
sigma_x=2
sigma_a=3
mean_a=1
generateRandomMatrix <- function(K, D) {
  randomVector <- sample(0:1, K * D, replace = TRUE)
  randomMatrix <- matrix(randomVector, nrow = K, ncol = D)
  return(randomMatrix)
}

Z_true <- generateRandomMatrix(N, K)
A_ <- matrix(rnorm(K*D,mean_a,sigma_a), nrow = K, ncol = D)
X_ <- matrix(0, nrow=N, ncol=D)
for (i in 1:N){
  X_[i,] <- mvrnorm(1, mu = Z_true[i,]%*%A_ , Sigma = sigma_x^2*diag(D) )

  
}


# Supponendo che A_ e X_ siano oggetti R di tipo matrice o data frame
result <- GibbsSampler_betabernoulli(alpha=-7, theta=16, sigma_x=sigma_x, sigma_a=sigma_a,
                                     n_tilde=30, N, A_=A_, X_=X_, n_iter=20, initial_iters=1000)


#prior_variance_sigma_x = 1
#prior_variance_sigma_a = 1
#GibbsSampler_betabernoulli( alpha = -7, theta = 16, sigma_x 0 sigma_x,sigma_a = sigma_a,prior_variance_sigma_x = prior_variance_sigma_x, 
#    prior_variance_sigma_a = prior_variance_sigma_a,  n_tilde = 30, n = N, A_ = A_, X_ = X_, n_iter = 20, initial_iters = 1000)


