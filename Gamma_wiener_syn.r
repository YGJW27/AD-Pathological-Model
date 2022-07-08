library("rstan")
library("sigmoid")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

generate_data <- function(NS, N, k_pos, k_neg, lambda_pos, lambda_neg,
                          omega_pos, omega_neg, omega_d, omega_0, c_mu, sigma, 
                          c_ga, beta, h_0, h_d, h_m, sigma_ep){
  N_vec <- c()
  T_total <- c()
  y_total <- c()
  effect_pos_total <- c()
  effect_neg_total <- c()
  
  mu_total <- c()
  alpha_total <- c()
  m_total <- c()
  dm_total <- c()
  d_total <- c()
  dd_total <- c()
  for (i in 1:NS){
    T <- seq(0, N-1, by=1)
    effect_count_pos <- rep(0, N-1)
    effect_count_neg <- rep(0, N-1)
    kt_pos <- rep(0, N-1)
    kt_neg <- rep(0, N-1)
    effect_pos <- rep(0, N-1)
    effect_neg <- rep(0, N-1)

    for (n in 1:(N-1)){
      kt_pos[n] <- k_pos * (T[n+1]-T[n])
      kt_neg[n] <- k_neg * (T[n+1]-T[n])
      effect_count_pos[n] <- rpois(1, kt_pos[n])
      effect_count_neg[n] <- rpois(1, kt_neg[n])
      effect_pos[n] <- rgamma(1, effect_count_pos[n], lambda_pos)
      effect_neg[n] <- rgamma(1, effect_count_neg[n], lambda_neg)
    }
    
    mu <- rep(0, N-1)
    alpha <- rep(0, N-1)
    
    m <- rep(0, N)
    dm <- rep(0, N-1)
    d <- rep(0, N)
    dd <- rep(0, N-1) 
    
    for (n in 1:(N-1)){
      mu[n] <- c_mu * tanh(omega_pos * effect_pos[n] - omega_neg * effect_neg[n]
                           + omega_d * d[n] + omega_0) * (T[n+1]-T[n])
      alpha[n] <- c_ga * sigmoid(m[n]) * (T[n+1]-T[n])
      dm[n] <- rnorm(1, mu[n], sigma)
      m[n+1] <- m[n] + dm[n]
      dd[n] <- rgamma(1, alpha[n], beta)
      d[n+1] <- d[n] + dd[n]
    }
    
    y <- rep(0, N)
    for (n in 1:N){
      y[n] <- h_0 + h_d * d[n] + h_m * m[n] + rnorm(1, 0, sigma_ep)
    }
    
    N_vec <- append(N_vec, N)
    T_total <- append(T_total, T)
    y_total <- append(y_total, y)
    effect_pos_total <- append(effect_pos_total, effect_pos)
    effect_neg_total <- append(effect_neg_total, effect_neg)
    
    mu_total <- c(mu_total, mu)
    alpha_total <- c(alpha_total, alpha)
    m_total <- c(m_total, m)
    dm_total <- c(dm_total, dm)
    d_total <- c(d_total, d)
    dd_total <- c(dd_total, dd)
  }
  list(N_vec=N_vec, T_total=T_total, y_total=y_total, 
       effect_pos_total=effect_pos_total, effect_neg_total=effect_neg_total,
       mu_total=mu_total, alpha_total=alpha_total, m_total=m_total,
       dm_total=dm_total, d_total=d_total, dd_total=dd_total)
}

N <- 21
k_pos <- 5
k_neg <- 5
lambda_pos <- 3
lambda_neg <- 3
omega_pos <- 0.1
omega_neg <- 0.1
omega_d <- 0.05
omega_0 <- 0
c_mu <- 3
sigma <- 0.01
c_ga <- 10
beta <- 10

h_0 <- 30
h_d <- -1
h_m <- -0.5
sigma_ep <- 0.5

NS <- 2

syn_data <- generate_data(NS, N, k_pos, k_neg, lambda_pos, lambda_neg,
                          omega_pos, omega_neg, omega_d, omega_0, c_mu, sigma,
                          c_ga, beta, h_0, h_d, h_m, sigma_ep)


model.file <- "D:/Documents/R/Gamma_wiener.stan"
ad_dat <- list(N_sample = NS, N_vec = syn_data$N_vec, 
               N_total = NROW(syn_data$T_total), T = syn_data$T_total, 
               y = syn_data$y_total, effect_pos=syn_data$effect_pos_total, 
               effect_neg=syn_data$effect_neg_total)

ad_init <- function() {
  list()
}

fit <- stan(file = model.file, data = ad_dat, chains = 2, iter=5000)

print(fit, pars=list("mu", "m", "mu_ob", "dm"), include=TRUE)
print(fit, pars=list("mu", "m", "mu_ob", "dm"), include=FALSE)
print(fit, pars=list("m"), include=TRUE)
plot(fit)
traceplot(fit, pars=list("omega_pos","omega_neg"), include=TRUE)
pairs(fit, pars = c("mu", "tau", "lp__"))