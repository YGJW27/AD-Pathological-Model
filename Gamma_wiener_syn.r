library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

generate_data <- function(NS, N, k_pos, k_neg, lambda_pos, lambda_neg,
                          omega_pos, omega_neg, omega_0, c_mu, sigma,
                          h_0, h_1, sigma_ep){
  N_vec <- c()
  T_total <- c()
  y_total <- c()
  effect_pos_total <- c()
  effect_neg_total <- c()
  for (i in 1:NS){
    T <- seq(0, N-1, by=1)
    effect_count_pos <- rep(0, N-1)
    effect_count_neg <- rep(0, N-1)
    kt_pos <- rep(0, N-1)
    kt_neg <- rep(0, N-1)
    effect_pos <- rep(0, N-1)
    effect_neg <- rep(0, N-1)
    mu <- rep(0, N-1)
    
    for (n in 1:(N-1)){
      kt_pos[n] <- k_pos * (T[n+1]-T[n])
      kt_neg[n] <- k_neg * (T[n+1]-T[n])
      effect_count_pos[n] <- rpois(1, kt_pos[n])
      effect_count_neg[n] <- rpois(1, kt_neg[n])
      effect_pos[n] <- rgamma(1, effect_count_pos[n], lambda_pos)
      effect_neg[n] <- rgamma(1, effect_count_neg[n], lambda_neg)
      mu[n] <- c_mu * tanh(omega_pos * effect_pos[n] - omega_neg * effect_neg[n] + omega_0) * (T[n+1]-T[n])
    }
    
    m <- rep(0, N)
    dm <- rep(0, N-1)
    for (n in 1:(N-1)){
      dm[n] <- rnorm(1, mu[n], sigma)
      m[n+1] <- m[n] + dm[n]
    }
    
    y <- rep(0, N)
    for (n in 1:N){
      y[n] <- h_0 + h_1 * m[n] + rnorm(1, 0, sigma_ep)
    }
    
    N_vec <- append(N_vec, N)
    T_total <- append(T_total, T)
    y_total <- append(y_total, y)
    effect_pos_total <- append(effect_pos_total, effect_pos)
    effect_neg_total <- append(effect_neg_total, effect_neg)
  }
  list(N_vec, T_total, y_total, effect_pos_total, effect_neg_total)
}

N <- 101
k_pos <- 5
k_neg <- 5
lambda_pos <- 3
lambda_neg <- 3
omega_pos <- 3
omega_neg <- 1.5
omega_0 <- -1
c_mu <- 0.5
sigma <- 0.01

h_0 <- 10
h_1 <- 1
sigma_ep <- 0.01

NS <- 2
syn_data <- generate_data(NS, N, k_pos, k_neg, lambda_pos, lambda_neg,
                          omega_pos, omega_neg, omega_0, c_mu, sigma,
                          h_0, h_1, sigma_ep)


model.file <- "D:/Documents/R/Gamma_wiener.stan"
ad_dat <- list(N_sample = NS, N_vec = unlist(syn_data[1]), 
               N_total = NROW(unlist(syn_data[2])), T = unlist(syn_data[2]), 
               y = unlist(syn_data[3]), effect_pos=unlist(syn_data[4]), 
               effect_neg=unlist(syn_data[5]))

fit <- stan(file = model.file, data = ad_dat, chains = 2, iter=5000)

print(fit, pars=list("mu", "m", "mu_ob", "dm"), include=FALSE)
print(fit, pars=list("m"), include=TRUE)
plot(fit)
traceplot(fit, pars=list("omega_pos","omega_neg"), include=TRUE)
pairs(fit, pars = c("mu", "tau", "lp__"))