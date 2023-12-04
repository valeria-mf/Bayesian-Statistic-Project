install.packages("Rcpp")
install.packages("RcppEigen")

library(Rcpp)
library(RcppEigen)

Sys.setenv("PKG_CXXFLAGS"="-I\"C:\\Users\\ipval\\Downloads\\eigen-3.4.0\\eigen-3.4.0\"")
Sys.setenv(PKG_CXXFLAGS=paste0("-I", shQuote("C:/Users/ipval/AppData/Local/R/win-library/4.2/RcppEigen/include")))

Rcpp::sourceCpp("C:/Users/ipval/Downloads/GibbsSamplerIBP.cpp")

n=5
A_ <- matrix(rnorm(4*5,1,3), nrow = 4, ncol = 5)
X_ <- matrix(rnorm(n*5,2,2), nrow = n, ncol = 5)


result <- GibbsSampler_IBP(alpha=0.7,gamma=2, sigma_a=3, 
                  sigma_x=2, theta=16, n=n, A_=A_, X_=X_, n_iter=20,initial_iters=0)
