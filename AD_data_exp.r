library("bayesplot")
library("ggplot2")
library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


data <- read.table("D:/data_for_hsmm_ADNI_220416.csv", sep = ",", header = TRUE)
DX_index <- apply(data, 1, function(r) r["X"] == "DX")
APOE4_index <- apply(data, 1, function(r) r["X"] == "APOE4")
MMSE_index <- apply(data, 1, function(r) r["X"] == "MMSE")

DX_data <- data[DX_index,]
RID <- DX_data["RID"]
data <- data[,-1][,-1]

MMSE_data <- data[MMSE_index,]
APOE4_data <- data[APOE4_index,]
APOE4_list <- APOE4_data[,1]

MMSE_data[MMSE_data == -1] <- NA
APOE4_data[APOE4_data == -1] <- NA
APOE4_list[APOE4_list == -1] <- NA

# delete row without APOE4
MMSE_data <- MMSE_data[!is.na(APOE4_list),]
APOE4_list <- APOE4_list[!is.na(APOE4_list)]

MMSE_total <- c()
T_total <- c()
N_vec <- c()
NS <- 0
z_1 <- c()
for (n in 1:NROW(APOE4_list)){
  idx <- !is.na(MMSE_data[n,])
  N_s <- NROW(colnames(idx)[idx])
  if (N_s >= 10){
    MMSE_total <- append(MMSE_total, MMSE_data[n,][idx])
    T_total <- append(T_total, as.integer(as.double(substring(colnames(idx)[idx], 2))))
    z_1 <- append(z_1, APOE4_list[n])
    N_vec <- append(N_vec, N_s)
    NS <- NS + 1
  }
}


model.file <- "D:/Documents/R/AD_Gamma_wiener.stan"
ad_dat <- list(N_sample = NS, N_vec = N_vec, 
               N_total = NROW(T_total), T = T_total/6, 
               y = MMSE_total, z_1 = z_1)

fit <- stan(file = model.file, data = ad_dat, chains = 2, 
            iter = 10000, warmup = 6000,
            control=list(max_treedepth=12, adapt_delta=0.8))

print(fit, 
      pars=list("omega_0", "omega_1", "omega_d", "c_mu", "sigma", 
                "c_ga", "beta", "h_0", "h_d", "h_m", "sigma_ep"), 
      include=TRUE)

mcmc_dens_overlay(
  fit,
  pars = c("omega_0", "omega_1", "omega_d", "c_mu", "sigma", 
           "c_ga", "beta", "h_0", "h_d", "h_m", "sigma_ep"),
  prob = 0.8, # 80% intervals
  prob_outer = 0.99, # 99%
  point_est = "mean"
)

print(fit, 
      pars=list("d"), 
      include=TRUE)
print(fit, 
      pars=list("m"), 
      include=TRUE)
print(fit, 
      pars=list("dd"), 
      include=TRUE)
print(fit, 
      pars=list("dm"), 
      include=TRUE)
print(fit, 
      pars=list("m_0"), 
      include=TRUE)
print(fit, 
      pars=list("d_0"), 
      include=TRUE)
