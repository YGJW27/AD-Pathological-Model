data {
  int<lower=0> N_sample;
  int<lower=0> N_vec[N_sample];
  int<lower=0> N_total;
  int<lower=0> T[N_total];
  real y[N_total];
  real z_1[N_sample];
}

transformed data {
}

parameters {
  real omega_0;
  real omega_1;
  real omega_d;
  real c_mu;
  real<lower=0.009,upper=0.011> sigma;
  real m_0[N_sample];
  real dm[N_total - N_sample];

  real<lower=0> c_ga;
  real<lower=0> beta;
  real d_0[N_sample];
  real<lower=0> dd[N_total - N_sample];

  real h_0;
  real h_d;
  real h_m;
  real<lower=0> sigma_ep;
}

transformed parameters {
  real m[N_total];
  real mu[N_total - N_sample];
  real sigma_t[N_total - N_sample];
  real d[N_total];
  real alpha[N_total - N_sample];
  real mu_ob[N_total];

  {
    int N_pre;
    N_pre = 0;
    for (idx in 1:N_sample){
      m[N_pre+1] = m_0[idx];
      d[N_pre+1] = d_0[idx];
      for (n in (N_pre+1):(N_pre+N_vec[idx]-1)){
        m[n+1] = m[n] + dm[n-idx+1];
        mu[n-idx+1] = c_mu * tanh(omega_1 * z_1[idx] + omega_d * d[n] + omega_0) * (T[n+1] - T[n]);
        sigma_t[n-idx+1] = sigma * sqrt(T[n+1] - T[n]);
        d[n+1] = d[n] + dd[n-idx+1];
        alpha[n-idx+1] = c_ga * inv_logit(m[n]) * (T[n+1] - T[n]);
      }
      N_pre = N_pre + N_vec[idx];
    }
  }
  
  for (n in 1:N_total)
    mu_ob[n] = h_0 + d[n] * h_d + m[n] * h_m;
}

model {
  omega_1 ~ normal(1, 0.5);
  omega_d ~ normal(0.5, 1);
  omega_0 ~ normal(0.5, 1);
  c_mu ~ normal(3, 5);
  sigma ~ normal(0.01, 0.1);
  c_ga ~ normal(2, 5);
  beta ~ normal(3, 5);

  h_0 ~ normal(28.5, 1);
  h_d ~ normal(-1, 5);
  h_m ~ normal(-0.5, 5);
  sigma_ep ~ normal(1.2, 1);

  m_0 ~ normal(0, 10);
  d_0 ~ normal(0, 10);

  for (n in 1:(N_total - N_sample)){
    dm[n] ~ normal(mu[n], sigma_t[n]);
    dd[n] ~ gamma(alpha[n], beta);
  }

  for (n in 1:N_total)
    y[n] ~ normal(mu_ob[n], sigma_ep);
}


