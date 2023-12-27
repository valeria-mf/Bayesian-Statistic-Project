# Install necessary packages if they are not already installed
# install.packages("Rcpp")
# install.packages("RcppEigen")

library(Rcpp)
library(RcppEigen)
library(MASS)  # For the mvrnorm function to generate multivariate normal samples

# Source the C++ file containing the Gibbs Sampler function
Rcpp::sourceCpp("/Users/antonionapolitano/Desktop/bayesian_mycode/GibbsSamplerBB.cpp")  # Update this path accordingly

# Set a seed for reproducibility
set.seed(1234)

# Define the parameters for the synthetic data generation
N <- 5  # Number of objects
K <- 3   # Number of features
D <- 3   # Dimensionality of the data
sigma_x <- 1
sigma_a <- 1
mean_a <- 0

# Function to generate a random binary matrix
generateRandomMatrix <- function(N, K) {
  randomVector <- sample(0:1, N * K, replace = TRUE)
  randomMatrix <- matrix(randomVector, nrow = N, ncol = K)
  return(randomMatrix)
}

# Generate the true latent feature matrix Z
Z_true <- generateRandomMatrix(N, K)

# Display the matrix Z_true 
print("Z_true matrix:")
print(Z_true)

# Generate a random matrix A
A_ <- matrix(rnorm(K * D, mean_a, sigma_a), nrow = K, ncol = D)

# Display the matrix A_ 
print("A_ matrix:")
print(A_)

# Generate the observed data matrix X
X_ <- matrix(0, nrow = N, ncol = D)
for (i in 1:N) {
  X_[i, ] <- mvrnorm(1, mu = Z_true[i, ] %*% A_, Sigma = sigma_x^2 * diag(D))
}

# Display the matrix X_ 
print("X_ matrix:")
print(X_)

# Running the Gibbs Sampler
result <- GibbsSampler_betabernoulli(alpha = -1, theta = 3, sigma_x = sigma_x,
                                     sigma_a = sigma_a, n_tilde = 3 , n = N,
                                     A = A_, X = X_, n_iter = 3, initial_iters = 1)


# You can add code here to analyze 'result', which should contain the output from your Gibbs sampler
