data {
  int<lower=0> N_total;
  int<lower=0> T[N_total];
  real<lower=0> effect_pos[N_total - 1];
  real<lower=0> effect_neg[N_total - 1];
  
  int N_obs;
  int idx_obs[N_obs];
  int idx_mis[N_total - N_obs];
  real y_obs[N_obs];
}

transformed data {
}

parameters {
  real<lower=0, upper=30> y_mis[N_total - N_obs];
  real omega_0;
  real<lower=0> omega_pos;
  real<upper=0> omega_neg;
  real omega_d;
  real c_mu;
  real<lower=0> sigma;
  real m_0;
  real dm[N_total - 1];

  real<lower=0> c_ga;
  real<lower=0> beta;
  real d_0;
  real<lower=0> dd[N_total - 1];

  real h_0;
  real h_d;
  real h_m;
  real<lower=0> sigma_ep;
}

transformed parameters {
  real y[N_total];
  real m[N_total];
  real mu[N_total - 1];
  real sigma_t[N_total - 1];
  real d[N_total];
  real alpha[N_total - 1];
  real mu_ob[N_total];

  y[idx_obs] = y_obs;
  y[idx_mis] = y_mis;
  
  {
    m[1] = m_0;
    d[1] = d_0;
    for (n in 1:(N_total - 1)){
      m[n+1] = m[n] + dm[n];
      mu[n] = c_mu * tanh(omega_pos * effect_pos[n]
            + omega_neg * effect_neg[n] + omega_d * d[n] + omega_0) * (T[n+1] - T[n]);
      sigma_t[n] = sigma * sqrt(T[n+1] - T[n]);
      d[n+1] = d[n] + dd[n];
      alpha[n] = c_ga * inv_logit(m[n]) * (T[n+1] - T[n]);
    }
  }
  
  for (n in 1:N_total)
    mu_ob[n] = h_0 + d[n] * h_d + m[n] * h_m;
}

model {
  omega_pos ~ normal(0.1, 0.1);
  omega_neg ~ normal(-0.1, 0.1);
  omega_d ~ normal(0.05, 0.1);
  omega_0 ~ normal(0, 0.1);
  c_mu ~ normal(3, 0.1);
  sigma ~ normal(0.5, 0.1);
  c_ga ~ normal(5, 0.5);
  beta ~ normal(10, 0.1);

  h_0 ~ normal(30, 0.1);
  h_d ~ normal(-1, 0.1);
  h_m ~ normal(-0.5, 0.1);
  sigma_ep ~ normal(0.5, 0.1);

  m_0 ~ normal(0, 10);
  d_0 ~ normal(0, 10);

  for (n in 1:(N_total - 1)){
    dm[n] ~ normal(mu[n], sigma_t[n]);
    dd[n] ~ gamma(alpha[n], beta);
  }

  for (n in 1:N_total)
    y[n] ~ normal(mu_ob[n], sigma_ep);
}

generated quantities {
}

