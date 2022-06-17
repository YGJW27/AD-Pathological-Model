library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)



N <- 101
T <- seq(0, N-1, by=1)
k_pos <- 5
k_neg <- 5
lambda_pos <- 3
lambda_neg <- 3
omega_pos <- 3
omega_neg <- 1.5
omega_0 <- -1
effect_count_pos <- rep(0, N-1)
effect_count_neg <- rep(0, N-1)
kt_pos <- rep(0, N-1)
kt_neg <- rep(0, N-1)
effect_pos <- rep(0, N-1)
effect_neg <- rep(0, N-1)
mu <- rep(0, N-1)
c_mu <- 0.5
sigma <- 0.01

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
h_0 <- 10
h_1 <- 1
sigma_ep <- 0.01
for (n in 1:N){
  y[n] <- h_0 + h_1 * m[n] + rnorm(1, 0, sigma_ep)
}


model.file <- "C:/Users/shear/Documents/R/Gamma_wiener.stan"
ad_dat <- list(N = N, T = T, y = y, effect_pos=effect_pos, effect_neg=effect_neg)
fit <- stan(file = model.file, data = ad_dat, chains = 2, iter=20000)

print(fit, pars=list("mu","omega_neg"), include=TRUE)
plot(fit)
traceplot(fit, pars=list("omega_pos","omega_neg"), include=TRUE)
pairs(fit, pars = c("mu", "tau", "lp__"))