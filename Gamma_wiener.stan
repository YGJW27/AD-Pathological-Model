
data {
  int<lower=0> N;
  int<lower=0> T[N];
  real y[N];
  real<lower=0> effect_pos[N-1];
  real<lower=0> effect_neg[N-1];
}

transformed data {
}

parameters {
  real omega_0;
  real<lower=0> omega_pos;
  real<lower=0> omega_neg;
  real c_mu;
  
  real m[N];
  real<lower=0> sigma;
  
  real h_0;
  real h_1;
  real<lower=0> sigma_ep;
}

transformed parameters {
  real mu[N-1];
  real mu_ob[N];
  real dm[N-1];

  for (n in 1:(N-1)){
    dm[n] = m[n+1] - m[n];
  }
  
  for (n in 1:N-1){
    mu[n] = c_mu * tanh(omega_pos * effect_pos[n]
            - omega_neg * effect_neg[n] + omega_0) * (T[n+1] - T[n]);
  }
  
  for (n in 1:N)
    mu_ob[n] = h_0 + m[n] * h_1;
}

model {
  omega_pos ~ normal(3, 0.01);
  omega_neg ~ normal(1.5, 0.01);
  omega_0 ~ normal(-1, 0.01);
  h_0 ~ normal(10, 0.01);
  h_1 ~ normal(1, 0.01);
  
  m[1] ~ normal(0, 0.01);
  
  for (n in 1:(N-1))
    dm[n] ~ normal(mu[n], sigma);
  
  for (n in 1:N)
    y[n] ~ normal(mu_ob[n], sigma_ep);
}


