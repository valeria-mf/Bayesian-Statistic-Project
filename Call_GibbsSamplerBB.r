install.packages("Rcpp")
install.packages("RcppEigen")

library(Rcpp)
library(RcppEigen)

#Questo è il path del file header contenuto nella cartella unzippata dell'installazione di Eigen
Sys.setenv("PKG_CXXFLAGS"="-I\"C:\\Users\\ipval\\Downloads\\eigen-3.4.0\\eigen-3.4.0\"")

#Questo è il path del file creato con il comando install.packages("RcppEigen")
Sys.setenv(PKG_CXXFLAGS=paste0("-I", shQuote("C:/Users/ipval/AppData/Local/R/win-library/4.2/RcppEigen/include")))

Rcpp::sourceCpp("C:/Users/ipval/Downloads/GibbsSamplerBB.cpp")

n=5
A_ <- matrix(rnorm(4*5,1,3), nrow = 4, ncol = 5)
X_ <- matrix(rnorm(n*5,2,2), nrow = n, ncol = 5)


# Supponendo che A_ e X_ siano oggetti R di tipo matrice o data frame
result <- GibbsSampler_betabernoulli(alpha=-7, theta=16, sigma_x=2, sigma_a=3,
                                     n_tilde=30, n, A_=A_, X_=X_, n_iter=20, initial_iters=0)

