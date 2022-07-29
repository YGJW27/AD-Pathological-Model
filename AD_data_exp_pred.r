library("bayesplot")
library("ggplot2")
library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

setting=5
set.seed(setting)
set.seed(9)
# data preparation
{
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
}

model.file <- "D:/Documents/R/AD_Gamma_wiener_pred.stan"

T_max <- max(T_total/6)

start_pred_idx <- 2
y_df <- data.frame(matrix(nrow=NS, ncol=T_max))
mean_df <- data.frame(matrix(nrow=NS, ncol=T_max))
ten_df <- data.frame(matrix(nrow=NS, ncol=T_max))
ninety_df <- data.frame(matrix(nrow=NS, ncol=T_max))

# for (s in 1: NS){
#   start_idx <- which(T_total == 0)[s]
#   T_total1 <- T_total[start_idx:(start_idx+N_vec[s]-1)]
#   y_df[s, T_total1/6+1] <- MMSE_total[start_idx:(start_idx+N_vec[s]-1)]
# }
# print(y_df)
# write.csv(y_df, "D:/AD_y_10.csv")

test_idx <- c(3, 4, 7, 9, 10, 13, 16, 19, 25, 27, 29, 32, 36, 40, 45, 46,
          51, 55, 58, 61, 73, 75, 84, 85, 90, 97, 100, 110)

for (s in test_idx){
  start_idx <- which(T_total == 0)[s]
  T_total1 <- T_total[start_idx:(start_idx+N_vec[s]-1)]
  y_df[s, T_total1/6+1] <- MMSE_total[start_idx:(start_idx+N_vec[s]-1)]
  print(y_df)
  for (n in start_pred_idx:N_vec[s]){
    y_obs <- MMSE_total[start_idx:(start_idx+n-2)]
    z_11 <- z_1[s]
    N_obs <- n - 1
    idx_obs <- seq(1, n-1)
    idx_mis <- seq(n, N_vec[s])
    
    ad_dat <- list(N_total=N_vec[s], T=T_total1/6, z_1=z_11, N_obs=N_obs, 
                   idx_obs=array(idx_obs, dim=length(idx_obs)), 
                   idx_mis=array(idx_mis, dim=length(idx_mis)), 
                   y_obs=array(y_obs, dim=length(y_obs)))
    message("s:", s, " n:", n)
    fit <- stan(file = model.file, data = ad_dat, chains = 1, 
                iter = 6000, warmup = 5000,
                control=list(max_treedepth=10, adapt_delta=0.8))
    fit_summary <- summary(fit, pars=list("y"), probs=c(0.05, 0.95))$summary
    mean_df[s, T_total1[n]/6+1] <- fit_summary[n, "mean"]
    ten_df[s, T_total1[n]/6+1] <- fit_summary[n, "5%"]
    ninety_df[s, T_total1[n]/6+1] <- fit_summary[n, "95%"]

    print(mean_df)
  }
}

write.csv(mean_df, "D:/AD_mean.csv")
write.csv(ten_df, "D:/AD_5.csv")
write.csv(ninety_df, "D:/AD_95.csv")

write.csv(y_df, "D:/AD_y.csv")

save.image("D:/ad_workspace.RData")

y_pred <- read.csv("D:/AD_mean.csv",)[,-1]
y_true <- read.csv("D:/AD_y.csv")[,-1]
error <- (y_true - y_pred)^2
error <- as.numeric(unlist(error))
mean(error, na.rm=TRUE)
7.168197 7.131406 7.178152 7.098574