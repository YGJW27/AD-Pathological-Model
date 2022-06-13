
data {
  int<lower=0> N;
  int<lower=0> T[N];
  real y[N];
  int<lower=0> pois_max;
}

transformed data {
}

parameters {
  real<lower=0> k_pos;
  real<lower=0> k_neg;
  real<lower=0> lambda_pos;
  real<lower=0> lambda_neg;
  real<lower=0> effect_pos[N-1];
  real<lower=0> effect_neg[N-1];
  real m[N];
  real<lower=0> d[N];
  
  real omega_0;
  real<lower=0> omega_pos;
  real<upper=0> omega_neg;
  real c_mu;

  real<lower=0> sigma;
  real mu_0;
  real<lower=0> sigma_0;
  
  real h_1;
  real<lower=0> sigma_ep;
}

transformed parameters {
  real<lower=0> kt_pos[N-1];
  real<lower=0> kt_neg[N-1];
  real dm[N-1];
  real dd[N-1];
  real mu[N-1];
  real mu_ob[N];
  
  real lp_pos[pois_max, N-1];
  real lp_neg[pois_max, N-1];
  
  for (n in 1:(N-1)){
    kt_pos[n] = k_pos * (T[n+1] - T[n]);
    kt_neg[n] = k_neg * (T[n+1] - T[n]);
    dm[n] = m[n+1] - m[n];
    dd[n] = d[n+1] - d[n];
  }
    
  for (n in 1:N-1){
    mu[n] = c_mu * tanh(omega_pos * effect_pos[n] 
            + omega_neg * effect_neg[n] + omega_0) * (T[n+1] - T[n]);
  }
  
  for (p in 1:pois_max)
    for(n in 1:N-1){
      lp_pos[p, n] = gamma_lpdf(effect_pos[n] | p, lambda_pos) + poisson_lpmf(p | kt_pos[n]);
      lp_neg[p, n] = gamma_lpdf(effect_neg[n] | p, lambda_neg) + poisson_lpmf(p | kt_neg[n]);
    }
  
  for (n in 1:N)
    mu_ob[n] = m[n] * h_1;
}

model {
  for (n in 1:N-1){
    target += log_sum_exp(lp_pos[1:pois_max, n]);
    target += log_sum_exp(lp_neg[1:pois_max, n]);
  }
  
  m[1] ~ normal(mu_0, sigma_0);
  dm ~ normal(mu, sigma);
  
  for (n in 1:N)
    y[n] ~ normal(mu_ob[n], sigma_ep);
}


