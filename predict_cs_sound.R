rm(list=ls())

# install required packages
options(repos='http://cran.rstudio.com/')
list.of.packages <- c('caret','mlbench', 'optparse', 'tuneR', 'seewave', 'pbapply', 'parallel', 'xgboost', 'randomForest', 'gbm')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,'Package'])]
if(length(new.packages))
  install.packages(new.packages, dependencies = TRUE)

library(caret)
library(mlbench)
library(optparse)
library(tuneR)
library(seewave)
library(xgboost)
library(dplyr)

data <- data.frame()
mp3_dir <- 'sound/tmp/'
wav_dir <- 'sound/proc/'
list <- list.files(mp3_dir, '\\mp3')
final_output <- data.frame()
file.remove("output_raw.csv")
file.remove("output_final.csv")
file.remove("speech-diarization.txt")

for (fileName in list) {
  row <- data.frame(fileName)
  data <- rbind(data, row)
}
names(data) <- c('sound.files')
sound.files <- as.character(unlist(data$sound.files))

lapp <- pbapply::pblapply

########################################################################################################
## convert MP3 to WAV with sample rate 16000

lapp(1:length(sound.files), function(i) {
  mp3_loc <- paste0(mp3_dir, sound.files[i])
  wav_loc <- paste0(wav_dir, sound.files[i])
  wav_loc <- gsub('.mp3', '.wav', wav_loc)
  #mp3 <- tuneR::readMP3(mp3_loc)
  #writeWave(mp3,wav_loc,extensible = FALSE)
  cmd <- paste('ffmpeg -i',mp3_loc,'-ar 16000','-ac 1 -ab 32k -y',wav_loc, sep = ' ')
  system(cmd, ignore.stdout = TRUE)
})

wav_data <- data.frame()
list <- list.files(wav_dir, '\\wav')
for (a in list) {
  row <- data.frame(a)
  wav_data <- rbind(wav_data, row)
}
names(wav_data) <- c('sound.files')
sound.files <- as.character(unlist(wav_data$sound.files))
output <- data.frame()

########################################################################################################
## extract acoustic parameters from sound: get from https://github.com/primaryobjects/voice-gender

bp <- c(0,22)
wl = 2048 
threshold = 5

for(i in sound.files) {
  
  wav_loc <- paste0("sound/proc/", i)
  wav <- tuneR::readWave(wav_loc, units = "seconds")
  
  b <- bp
  if(b[2] > ceiling(wav@samp.rate/2000) - 1) b[2] <- ceiling(wav@samp.rate/2000) - 1
  
  songspec <- seewave::spec(wav, f = wav@samp.rate, plot = FALSE)
  analysis <- seewave::specprop(songspec, f = wav@samp.rate, flim = c(0, 280/1000), plot = FALSE)
  
  meanfreq <- analysis$mean/1000
  sd <- analysis$sd/1000
  median <- analysis$median/1000
  Q25 <- analysis$Q25/1000
  Q75 <- analysis$Q75/1000
  IQR <- analysis$IQR/1000
  skew <- analysis$skewness
  kurt <- analysis$kurtosis
  sp.ent <- analysis$sh
  sfm <- analysis$sfm
  mode <- analysis$mode/1000
  centroid <- analysis$cent/1000
  
  peakf <- 0#seewave::fpeaks(songspec, f = wav@samp.rate, wl = wl, nmax = 3, plot = FALSE)[1, 1]
  ff <- seewave::fund(wav, f = wav@samp.rate, ovlp = 50, threshold = threshold, fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = wl)[, 2]
  meanfun<-mean(ff, na.rm = T)
  minfun<-min(ff, na.rm = T)
  maxfun<-max(ff, na.rm = T)
  
  y <- seewave::dfreq(wav, f = wav@samp.rate, wl = wl, ylim=c(0, 280/1000), ovlp = 0, plot = F, threshold = threshold, bandpass = b * 1000, fftw = TRUE)[, 2]
  meandom <- mean(y, na.rm = TRUE)
  mindom <- min(y, na.rm = TRUE)
  maxdom <- max(y, na.rm = TRUE)
  dfrange <- (maxdom - mindom)
  
  changes <- vector()
  for(j in which(!is.na(y))){
    change <- abs(y[j] - y[j + 1])
    changes <- append(changes, change)
  }
  if(mindom==maxdom) modindx<-0 else modindx <- mean(changes, na.rm = T)/dfrange
  
  label <- strsplit(i, "_")[[1]][1]
  
  x <- c(label, meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx)
  x <- data.frame(x)
  
  rownames(x) <- c("label", "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", "sfm","mode", "centroid", "peakf", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx")
  output <- rbind(output, t(x))
}

rownames(output) <- c(1:nrow(output))


########################################################################################################
## preparing the data

#get important variables
load("model/rfeResults.RDS")
impVars <- predictors(results)
output_results <- data.frame()

acoustics <- output %>% select(-label)
label <- output %>% select(label)
acoustics_new <- acoustics
acoustics_new[,colnames(acoustics)] <- sapply(acoustics_new[,colnames(acoustics)], as.character)
acoustics_new[,colnames(acoustics)] <- sapply(acoustics_new[,colnames(acoustics)], as.double)

acoustics_imp <- acoustics_new[,impVars]
acoustics_imp_w_label <- cbind(acoustics_imp, label)

new_output <- cbind(acoustics_new, output %>% select(label))

recipe_obj <- recipe(label ~., data = new_output) %>% step_scale(all_predictors(), -all_outcomes()) %>% prep(data = new_output)
recipe_obj

x_output <- bake(rec_obj, new_data = new_output) %>% select(-label)
y_output_vec <- ifelse(pull(new_output, label) == "female",1, 0)


########################################################################################################
## predict gender
#load models
load("model/genderRFE.RDS")
load("model/genderForest.RDS")
load("model/genderXgboost.RDS")
load("model/genderGBM.RDS")

GLM <- predict(genderRFE, newdata = acoustics_imp)
RandomForest <- predict(genderForest_2, newdata = acoustics_imp)
Xgboost <- predict(genderXgboost_2, newdata = acoustics_imp)
GBM <- predict(genderGBM_2, newdata = acoustics_imp)

model_keras <- load_model_hdf5("deeplearning_2.h5")

pred_class <- predict_classes(object = model_keras, x = as.matrix(x_output)) %>%
  as.vector()

pred_prob  <- predict_proba(object = model_keras, x = as.matrix(x_output)) %>%
  as.vector()

options(yardstick.event_first = FALSE)
estimates_res <- tibble(
  truth      = as.factor(y_output_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(pred_class) %>% fct_recode(yes = "1", no = "0"),
  class_prob = pred_prob
)
estimates_res %>% metrics(truth, estimate)
estimates_res %>% yardstick::precision(truth, estimate)
estimates_res %>% yardstick::recall(truth, estimate)

pred_class

output_results <- cbind(output %>% select(label), GLM, RandomForest, Xgboost, GBM)

glm_table <- table(output_results$GLM, output_results$label)
gbm_table <- table(output_results$GBM, output_results$label)
rf_table <- table(output_results$RandomForest, output_results$label)
xgb_table <- table(output_results$Xgboost, output_results$label)
confusionMatrix(glm_table)
confusionMatrix(gbm_table)
confusionMatrix(rf_table)
confusionMatrix(xgb_table)
table(pred_class, y_output_vec)

precision(glm_table)
precision(gbm_table)
precision(rf_table)
precision(xgb_table)

recall(glm_table)
recall(gbm_table)
recall(rf_table)
recall(xgb_table)

### glm
predictedGender_glm <- as.data.frame(predict(genderRFE, newdata = acoustics_imp_w_label, type = "prob")$female)
glm_pred_df <- cbind(label, predictedGender_glm)
colnames(glm_pred_df) <- c("truth", "class_prob")
glm_pred_df <- glm_pred_df %>% mutate(truth = truth %>% as.factor())

auc_glm <- roc_auc(glm_pred_df, truth, class_prob)

glm_roc_curve <- roc_curve(glm_pred_df, truth, class_prob)
autoplot(glm_roc_curve)

## rf
predictedGender_rf <- as.data.frame(predict(genderForest_2, newdata = acoustics_imp_w_label, type = "prob")$female)
rf_pred_df <- cbind(label, predictedGender_rf)
colnames(rf_pred_df) <- c("truth", "class_prob")
rf_pred_df <- rf_pred_df %>% mutate(truth = truth %>% as.factor())

auc_rf <- roc_auc(rf_pred_df, truth, class_prob)

rf_roc_curve <- roc_curve(rf_pred_df, truth, class_prob)
autoplot(rf_roc_curve)

## xgb
predictedGender_xgb <- as.data.frame(predict(genderXgboost_2, newdata = acoustics_imp_w_label, type = "prob")$female)
xgb_pred_df <- cbind(label, predictedGender_xgb)
colnames(xgb_pred_df) <- c("truth", "class_prob")
xgb_pred_df <- xgb_pred_df %>% mutate(truth = truth %>% as.factor())

auc_xgb <- roc_auc(xgb_pred_df, truth, class_prob)

xgb_roc_curve <- roc_curve(xgb_pred_df, truth, class_prob)
autoplot(xgb_roc_curve)

## gbm
predictedGender_gbm <- as.data.frame(predict(genderGBM_2, newdata = acoustics_imp_w_label, type = "prob")$female)
gbm_pred_df <- cbind(label, predictedGender_gbm)
colnames(gbm_pred_df) <- c("truth", "class_prob")
gbm_pred_df <- gbm_pred_df %>% mutate(truth = truth %>% as.factor())

auc_gbm <- roc_auc(gbm_pred_df, truth, class_prob)

gbm_roc_curve <- roc_curve(gbm_pred_df, truth, class_prob)
autoplot(gbm_roc_curve) 

##mlp

auc_mlp <- estimates_res %>% roc_auc(truth, class_prob)
mlp_roc_curve <- estimates_res %>% roc_curve(truth, class_prob)

autoplot(mlp_roc_curve)

#combined plot

plot1 <- autoplot(glm_roc_curve) + ggtitle("GLM") + annotation_custom(grid.text(paste0("AUC: ", round(auc_glm$.estimate, digits = 3)), x=0.8,  y=0.05, gp=gpar(fontsize=8)))

plot2 <- autoplot(rf_roc_curve) + ggtitle("Random Forest") + annotation_custom(grid.text(paste0("AUC: ", round(auc_rf$.estimate, digits = 3)), x=0.8,  y=0.05, gp=gpar(fontsize=8)))

plot3 <- autoplot(xgb_roc_curve) + ggtitle("XgBoost") + annotation_custom(grid.text(paste0("AUC: ", round(auc_xgb$.estimate, digits = 3)), x=0.8,  y=0.05, gp=gpar(fontsize=8)))

plot4 <- autoplot(gbm_roc_curve) + ggtitle("GBM") + annotation_custom(grid.text(paste0("AUC: ", round(auc_gbm$.estimate, digits = 3)), x=0.8,  y=0.05, gp=gpar(fontsize=8)))

plot5 <- autoplot(mlp_roc_curve) + ggtitle("MLP") + annotation_custom(grid.text(paste0("AUC: ", round(auc_mlp$.estimate, digits = 3)), x=0.8,  y=0.05, gp=gpar(fontsize=8)))

grid.arrange(plot1, plot2, plot3, plot4, plot5, ncol=3)






write.csv(output_results, file = "res.csv", col.names = TRUE, append = TRUE)

model_type.keras.models.Sequential <- function(x, ...) {
  "classification"
}

predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
}

explainer <- lime::lime(
  x              = x_output, 
  model          = model_keras, 
  bin_continuous = FALSE
)

explanation <- lime::explain(
  x_output[1:90,], 
  explainer    = explainer, 
  n_labels     = 1, 
  n_features   = 5,
  kernel_width = 0.5
)


plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Heatmap")

plot_features(explanation) +
  labs(title = "LIME Feature Importance Visualization",
       subtitle = "Hold Out (Test) Set, First 10 Cases Shown")

corrr_analysis <- x_output %>%
  mutate(label = y_output_vec) %>%
  correlate() %>%
  focus(label) %>%
  rename(feature = rowname) %>%
  arrange(abs(label)) %>%
  mutate(feature = as_factor(feature)) 
corrr_analysis

corrr_analysis %>%
  ggplot(aes(x = label, y = fct_reorder(feature, desc(label)))) +
  geom_point() +
  # Positive Correlations - Contribute to male
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[2]], 
               data = corrr_analysis %>% filter(label > 0)) +
  geom_point(color = palette_light()[[2]], 
             data = corrr_analysis %>% filter(label > 0)) +
  # Negative Correlations - contribute to female
  geom_segment(aes(xend = 0, yend = feature), 
               color = palette_light()[[1]], 
               data = corrr_analysis %>% filter(label < 0)) +
  geom_point(color = palette_light()[[1]], 
             data = corrr_analysis %>% filter(label < 0)) +
  # Vertical lines
  geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics
  theme_tq() +
  labs(title = "Gender Correlation Analysis",
       subtitle = paste("Positive Correlations (contribute to Female),",
                        "Negative Correlations (contribute to Male)"),
       y = "Feature Importance")

