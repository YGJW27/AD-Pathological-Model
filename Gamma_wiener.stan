data {
  int<lower=0> N_sample;
  int<lower=0> N_vec[N_sample];
  int<lower=0> N_total;
  int<lower=0> T[N_total];
  real y[N_total];
  real<lower=0> effect_pos[N_total - N_sample];
  real<lower=0> effect_neg[N_total - N_sample];
}

transformed data {
  
}

parameters {
  real omega_0;
  real<lower=0> omega_pos;
  real<lower=0> omega_neg;
  real c_mu;
  
  real m[N_total];
  real<lower=0> sigma;
  
  real h_0;
  real h_1;
  real<lower=0> sigma_ep;
}

transformed parameters {
  real mu[N_total - N_sample];
  real dm[N_total - N_sample];
  real mu_ob[N_total];
  
  {
    int N_pre;
    N_pre = 0;
    for (idx in 1:N_sample){
      for (n in (N_pre+1):(N_pre+N_vec[idx]-1)){
        dm[n-idx+1] = m[n+1] - m[n];
        mu[n-idx+1] = c_mu * tanh(omega_pos * effect_pos[n-idx+1]
            - omega_neg * effect_neg[n-idx+1] + omega_0) * (T[n+1] - T[n]);
      }
      N_pre = N_pre + N_vec[idx];
    }
  }
  
  for (n in 1:N_total)
    mu_ob[n] = h_0 + m[n] * h_1;
}

model {
  omega_pos ~ normal(3, 0.01);
  omega_neg ~ normal(1.5, 0.01);
  omega_0 ~ normal(-1, 0.01);
  h_0 ~ normal(10, 0.01);
  h_1 ~ normal(1, 0.01);
  
  {
    int N_pre;
    N_pre = 0;
    for (idx in 1:N_sample){
      m[N_pre+1] ~ normal(0, 0.01);
      N_pre = N_pre + N_vec[idx];
    }
  }

  for (n in 1:(N_total - N_sample))
    dm[n] ~ normal(mu[n], sigma);
  
  for (n in 1:N_total)
    y[n] ~ normal(mu_ob[n], sigma_ep);
}


