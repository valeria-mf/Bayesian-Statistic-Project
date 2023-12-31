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




plot_dual_graphs <- function(n_iter, initial_iters, result) {
  # Grafico di K
  plot(1:(n_iter + initial_iters), result$K_vector, 
       main = 'Number of latent features', xlab = 'Iteration', ylab = 'K+', type = 'l')
  abline(v = initial_iters, col = 'red')
  
  # Grafico log P(X,Z)
  plot(1:(n_iter + initial_iters), result$logPXZ_vector, 
       main = 'log P(X, Z)', xlab = 'Iteration', ylab = 'log P(X, Z)', type = 'l')
  abline(v = initial_iters, col = 'red')
}


plot_dual_graphs(n_iter, initial_iters, result)





library(ggplot2)
library(reshape2)
library(cowplot)

plot_dual_heatmaps <- function(A_, Expected_A) {
  
 
  df_expected_a1 <- melt(A_)
  df_expected_a2 <- melt(Expected_A)
  
  
  plot1 <- ggplot(df_expected_a1, aes(x = Var2, y = Var1, fill = value)) +
    geom_tile() +
    labs(title = "Original A",
         x = "Column Index",
         y = "Row Index") +
    scale_fill_gradient(low = "white", high = "blue", name = "Value") +
    theme_minimal()
  
  plot2 <- ggplot(df_expected_a2, aes(x = Var2, y = Var1, fill = value)) +
    geom_tile() +
    labs(title = "Expected A",
         x = "Column Index",
         y = "Row Index") +
    scale_fill_gradient(low = "white", high = "blue", name = "Value") +
    theme_minimal()
  
  combined_plot <- plot_grid(plot1, plot2,  ncol = 2)
  
  print(combined_plot)
}


plot_dual_heatmaps(A_, result$Expected_A)







library(ggplot2)
library(cowplot)

plot_combined_graphs <- function(n_iter, initial_iters, result) {
  # Creazione del dataframe per i dati
  data <- data.frame(iteration = 1:(n_iter + initial_iters),
                     K = result$K_vector,
                     logPXZ = result$logPXZ_vector)
  
  # Creazione dei grafici
  plot_k <- ggplot(data, aes(x = iteration, y = K)) +
    geom_line() +
    labs(title = 'Number of latent features',
         x = 'Iteration',
         y = 'K+') +
    theme_minimal() +
    geom_vline(xintercept = initial_iters, linetype = 'dashed', color = 'red')
  
  plot_logPXZ <- ggplot(data, aes(x = iteration, y = logPXZ)) +
    geom_line() +
    labs(title = 'log P(X, Z)',
         x = 'Iteration',
         y = 'log P(X, Z)') +
    theme_minimal() +
    geom_vline(xintercept = initial_iters, linetype = 'dashed', color = 'red')
  
  # Combina i due grafici sopra e sotto
  combined_plot <- plot_grid(plot_k, plot_logPXZ, nrow = 2)
  
  # Visualizza il grafico combinato
  print(combined_plot)
}

# Esempio di utilizzo della funzione con i parametri specificati
plot_combined_graphs(n_iter, initial_iters, result)







# Verificare se tutte le matrici nella lista sono identiche alla prima
sono_tutte_identiche <- all(sapply(result$Z_list[-1], function(matrice) identical(result$Z_list[[1]], matrice)))
cat("Sono tutte le matrici nella lista identiche alla prima?", sono_tutte_identiche, "\n")


#prior_variance_sigma_x = 1
#prior_variance_sigma_a = 1
#GibbsSampler_betabernoulli( alpha = -7, theta = 16, sigma_x 0 sigma_x,sigma_a = sigma_a,prior_variance_sigma_x = prior_variance_sigma_x, 
#    prior_variance_sigma_a = prior_variance_sigma_a,  n_tilde = 30, n = N, A_ = A_, X_ = X_, n_iter = 20, initial_iters = 1000)


