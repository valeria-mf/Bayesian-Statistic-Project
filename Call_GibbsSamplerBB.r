install.packages("Rcpp")
install.packages("RcppEigen")

library(Rcpp)
library(RcppEigen)
library(MASS)

#Questo è il path del file header contenuto nella cartella unzippata dell'installazione di Eigen
Sys.setenv("PKG_CXXFLAGS"="-I\"C:\\Users\\ipval\\Desktop\\Progetto Bayesiana\\eigen-3.4.0\\eigen-3.4.0\"")

#Questo è il path del file creato con il comando install.packages("RcppEigen")
Sys.setenv(PKG_CXXFLAGS=paste0("-I", shQuote("C:/Users/ipval/AppData/Local/R/win-library/4.2/RcppEigen/include")))

Rcpp::sourceCpp("C:/Users/ipval/Desktop/ProgettoBayesiana/GibbsSampler_betabernoulli.cpp")


set.seed(1234)

N=20
K=8
D=30
sigma_x=2
sigma_a=5
mean_a=1
n_iter=1000
initial_iters=100
generateRandomMatrix <- function(n_righe, n_colonne) {
  randomVector <- sample(0:1, n_righe * n_colonne, replace = TRUE)    #genera Z(i,j)=1 with P=0.5
  randomMatrix <- matrix(randomVector, nrow = n_righe, ncol = n_colonne)
  return(randomMatrix)
}

Z_true <- generateRandomMatrix(N, K)  
A_ <- matrix(rnorm(K*D,mean_a,sigma_a), nrow = K, ncol = D)
X_ <- matrix(0, nrow=N, ncol=D)
for (i in 1:N){
  X_[i,] <- mvrnorm(1, mu = Z_true[i,]%*%A_ , Sigma = sigma_x^2*diag(D) )

  
}
numero_colonne_diverse_da_zero_di_Z <- sum(colSums(Z_true != 0) > 0)
numero_colonne_diverse_da_zero_di_Z

result <- GibbsSampler_betabernoulli(alpha=-7, theta=13, sigma_x=sigma_x, sigma_a=sigma_a,
                                     n_tilde=30, n=N, A_=A_, X_=X_, n_iter=n_iter, initial_iters=initial_iters)
result$K_vector


par(mfrow=c(2,1))
#Grafico di K
plot(1:(n_iter +initial_iters), result$K_vector, main='Number of latent features',xlab='Iteration',ylab='K+',type='l')
abline(v=initial_iters, col='red')

#Grafico log P(X,Z)
plot(1:(n_iter +initial_iters), result$logPXZ_vector, main='log P(X, Z)',xlab='Iteration',ylab='log P(X, Z)',type='l')
abline(v=initial_iters, col='red')





# Verificare se tutte le matrici nella lista sono identiche alla prima
sono_tutte_identiche <- all(sapply(result$Z_list[-1], function(matrice) identical(result$Z_list[[1]], matrice)))
cat("Sono tutte le matrici nella lista identiche alla prima?", sono_tutte_identiche, "\n")


#prior_variance_sigma_x = 1
#prior_variance_sigma_a = 1
#GibbsSampler_betabernoulli( alpha = -7, theta = 16, sigma_x 0 sigma_x,sigma_a = sigma_a,prior_variance_sigma_x = prior_variance_sigma_x, 
#    prior_variance_sigma_a = prior_variance_sigma_a,  n_tilde = 30, n = N, A_ = A_, X_ = X_, n_iter = 20, initial_iters = 1000)


