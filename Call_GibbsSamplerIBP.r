install.packages("Rcpp")
install.packages("RcppEigen")

library(Rcpp)
library(RcppEigen)

Sys.setenv("PKG_CXXFLAGS"="-I\"C:\\Users\\ipval\\Downloads\\eigen-3.4.0\\eigen-3.4.0\"")
Sys.setenv(PKG_CXXFLAGS=paste0("-I", shQuote("C:/Users/ipval/AppData/Local/R/win-library/4.2/RcppEigen/include")))

Rcpp::sourceCpp("C:/Users/ipval/Downloads/GibbsSamplerIBP.cpp")

n=5
A_ <- matrix(1:20, nrow = 4, ncol = 5)
X_ <- matrix(1:25, nrow = n, ncol = 5)


result <- GibbsSampler_IBP(alpha=0.5,gamma=0.5, sigma_a=1, 
                  sigma_x=1, theta=0.6, n=n, A_=A_, X_=X_, n_iter=20,initial_iters=0)
