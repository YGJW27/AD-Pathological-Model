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
  real omega_d;
  real c_mu;
  real<lower=0.009,upper=0.011> sigma;
  real m[N_total];
  
  real d[N_total];
  real<lower=0> c_ga;
  real<lower=0> beta;

  real h_0;
  real h_d;
  real h_m;
  real<lower=0> sigma_ep;
}

transformed parameters {
  real mu[N_total - N_sample];
  real dm[N_total - N_sample];
  real alpha[N_total - N_sample];
  real<lower=0> dd[N_total - N_sample];
  real mu_ob[N_total];
  
  {
    int N_pre;
    N_pre = 0;
    for (idx in 1:N_sample){
      for (n in (N_pre+1):(N_pre+N_vec[idx]-1)){
        dm[n-idx+1] = m[n+1] - m[n];
        mu[n-idx+1] = c_mu * tanh(omega_pos * effect_pos[n-idx+1]
            - omega_neg * effect_neg[n-idx+1] + omega_d * d[n] + omega_0) * (T[n+1] - T[n]);
        dd[n-idx+1] = d[n+1] - d[n];
        alpha[n-idx+1] = c_ga * inv_logit(m[n]) * (T[n+1] - T[n]);
      }
      N_pre = N_pre + N_vec[idx];
    }
  }
  
  for (n in 1:N_total)
    mu_ob[n] = h_0 + d[n] * h_d + m[n] * h_m;
}

model {
  omega_pos ~ normal(0.1, 0.01);
  omega_neg ~ normal(0.1, 0.01);
  omega_d ~ normal(0.05, 0.001);
  omega_0 ~ normal(0, 0.01);
  c_mu ~ normal(3, 0.01);
  c_ga ~ normal(10, 0.01);
  beta ~ normal(10, 0.01);


  h_0 ~ normal(30, 0.01);
  h_d ~ normal(-1, 0.01);
  h_m ~ normal(-0.5, 0.01);

  
  {
    int N_pre;
    N_pre = 0;
    for (idx in 1:N_sample){
      m[N_pre+1] ~ normal(0, 0.01);
      d[N_pre+1] ~ normal(0, 0.01);
      N_pre = N_pre + N_vec[idx];
    }
  }

  for (n in 1:(N_total - N_sample)){
    dm[n] ~ normal(mu[n], sigma);
    dd[n] ~ gamma(alpha[n], beta);
  }
    
  
  for (n in 1:N_total)
    y[n] ~ normal(mu_ob[n], sigma_ep);
}


