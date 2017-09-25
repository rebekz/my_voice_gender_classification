#!/usr/bin/env Rscript

########################################################################################################
## get gender prediction from audio recording telephone conversation. Only supporting one to one conversation.
## Requirements:
## - Docker
## - R
## Author: Fitra Kacamarga

rm(list=ls())

options(warn=-1)

# install required packages
options(repos='http://cran.rstudio.com/')
list.of.packages <- c('caret','mlbench', 'optparse', 'tuneR', 'seewave', 'pbapply', 'parallel', 'xgboost', 'gbm')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,'Package'])]
if(length(new.packages))
  install.packages(new.packages, dependencies = TRUE)

library(caret)
library(mlbench)
library(optparse)
library(tuneR)
library(seewave)
library(xgboost)

option_list = list(
  make_option(c("-m", "--source"), type="character", default='sound/tmp/', 
              help="source audio file dir (in MP3/WAV format) [default= %default]", metavar="character"),
  make_option(c("-w", "--target"), type="character", default="sound/proc/", 
              help="target audio file dir [default= %default]", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="output_final.csv", 
              help="output filename [default= %default]", metavar="character"),
  make_option(c("-e", "--audio"), type="character", default=FALSE, 
              help="extracted from audio [default= %default]", metavar="boolean")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

mp3_dir <- opt$source
wav_dir <- opt$target
output_filename <- opt$out
extractFromSound = opt$audio

file.remove(output_filename)
file.remove("output_full.csv")
data <- data.frame()
list_1 <- list.files(mp3_dir, '\\mp3')
list_2 <- list.files(mp3_dir, '\\wav')

list <- c(list_1, list_2)

for (fileName in list) {
  row <- data.frame(fileName)
  data <- rbind(data, row)
}
names(data) <- c('sound.files')
sound.files <- as.character(unlist(data$sound.files))


########################################################################################################
## convert to WAV with sample rate 16000

convertToWav <- function(sound.files, mp3_dir = "sound/tmp/", wav_dir = "sound/proc/") {

  lapp <- pbapply::pblapply

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
  return(sound.files)
}


########################################################################################################
## get speech diarization from audio file. we used AaltoASR: https://github.com/aalto-speech/speaker-diarization

getSpeechDiarization <- function(sound.files) {
  
  system('docker run -d --name speech-diarizer blabbertabber/aalto-speech-diarizer tail -f /dev/null', ignore.stdout = TRUE)
  system("docker exec speech-diarizer /bin/bash -c 'mkdir /speaker-diarization/wav/'", ignore.stdout = TRUE)
  system("docker cp spk-diarization3.py speech-diarizer:/speaker-diarization/", ignore.stdout = TRUE)
  
  file.remove("speech-diarization.txt")
  
  for(i in 1:length(sound.files)) {
    sound_file <- sound.files[i]
    cmd <- paste0("docker cp sound/proc/",sound_file," speech-diarizer:/speaker-diarization/wav/")
    system(cmd, ignore.stdout = TRUE)
    cmd <- paste0("docker exec speech-diarizer /bin/bash -c 'cd /speaker-diarization/ && ./spk-diarization3.py wav/",sound_file,"'")
    system(cmd, ignore.stdout = TRUE)
    system("docker exec speech-diarizer /bin/bash -c 'cd /speaker-diarization/ && cat stdout' >> speech-diarization.txt", ignore.stdout = FALSE)
  }

  system("docker stop speech-diarizer", ignore.stdout = TRUE, ignore.stderr = TRUE)
  system("docker rm speech-diarizer", ignore.stdout = TRUE, ignore.stderr = TRUE)


  ########################################################################################################
  ## parse speech diarization output

  spk <- data.frame()

  text <- readLines("speech-diarization.txt",encoding="UTF-8")
  num_l <- length(text)
  p <- 0
  for(i in text) {
    proc <- p / num_l * 100
    print(paste0("Processing:", proc, " %"))
    text_split <- unlist(strsplit(i, " "))
    cst <- unlist(strsplit(text_split[1], "/"))[2]
    cst <- gsub("\\.wav", "", cst)
    
    str_time <- unlist(strsplit(text_split[3], "="))[2]
    fns_time <- unlist(strsplit(text_split[4], "="))[2]
    spker <- unlist(strsplit(text_split[5], "="))[2]
    
    row <- data.frame(cst, str_time, fns_time, spker)
    spk <- rbind(spk, row)
    p = p + 1
  }

  return(spk)

}


########################################################################################################
## extract acoustic parameters from audio: from https://github.com/primaryobjects/voice-gender

extractAcoustic <- function(spk, bp = c(0,22), wl = 2048, threshold = 5) {

  output <- data.frame()
  num_l <- nrow(spk)
  for(i in 1:nrow(spk)) {
    proc <- i / num_l * 100
    print(paste0("Processing:", proc, " %"))
    cst <- as.character(spk[i,1])
    str <- as.double(as.character(spk[i,2]))
    fin <- as.double(as.character(spk[i,3]))
    spker <- as.character(spk[i,4])
    
    wav_loc <- paste0("sound/proc/",cst, ".wav")
    wav <- tuneR::readWave(wav_loc, from = str, to = fin, units = "seconds")
    
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
    
    x <- c(cst, str, fin, spker, meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, centroid, peakf, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx)
    x <- data.frame(x)
    
    rownames(x) <- c("cst", "str_time", "fin_time", "spker", "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", "sfm","mode", "centroid", "peakf", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx")
    output <- rbind(output, t(x))
    
  }

  rownames(output) <- c(1:nrow(output))
  
  output[,5:25] <- sapply(output[,5:25], as.character)
  output[,5:25] <- sapply(output[,5:25], as.double)
  output[is.na(output)] <- 0

  write.csv(output, file = "acoustic_parameters.csv", col.names = TRUE, append = TRUE)

  return(output)
}

########################################################################################################
## predict gender

predictGender <- function(output) {

  #get important variables
  load("model/rfeResults.RDS")
  impVars <- predictors(results)
  output_results <- data.frame()

  for(i in 1:nrow(output)) {
    acoustics <- output[i, 5:25]
    cst <- output[i,1]
    str <- output[i,2]
    fin <- output[i,3]
    spker <- output[i,4]
    #used important variables
    acoustics_new <- acoustics[,impVars]
    acoustics_new[,impVars] <- sapply(acoustics_new[,impVars], as.character)
    acoustics_new[,impVars] <- sapply(acoustics_new[,impVars], as.double)
    
    #load models
    load("model/genderRFE.RDS")
    load("model/genderForest.RDS")
    load("model/genderXgboost.RDS")
    load("model/genderGBM.RDS")
    
    GLM <- predict(genderRFE, newdata = acoustics_new)
    RandomForest <- predict(genderForest_2, newdata = acoustics_new)
    Xgboost <- predict(genderXgboost_2, newdata = acoustics_new)
    GBM <- predict(genderGBM_2, newdata = acoustics_new)
    
    tmp_res <- cbind(output[i,], GLM, RandomForest, Xgboost, GBM)
    
    output_results <- rbind(output_results, tmp_res)
    
  }

  write.csv(output_results, file = "output_full.csv", col.names = TRUE, append = TRUE)
  return(output_results)

}

########################################################################################################
## Lets summarise

summarise <- function(output_results) {

  final_output <- data.frame()
  unique_pn <- unique(output_results$cst)
  for(i in unique_pn) {
    output_results$cst <- as.character(output_results$cst)
    cst_1 <- i
    subset <- subset(output_results, cst == cst_1)
    subset_cst <- subset(subset, spker == "speaker_1")
   
    #count number of female
    numFemale <- nrow(subset(subset_cst, GLM == "female")) + nrow(subset(subset_cst, RandomForest == "female")) + nrow(subset(subset_cst, Xgboost == "female")) + nrow(subset(subset_cst, GBM == "female"))
    ratioFemale <- numFemale / nrow(subset_cst) / 4
    #count number of male
    numMale <- nrow(subset(subset_cst, GLM == "male")) + nrow(subset(subset_cst, RandomForest == "male")) + nrow(subset(subset_cst, Xgboost == "male")) + nrow(subset(subset_cst, GBM == "male"))
    ratioMale <- numMale / nrow(subset_cst) / 4
    # print("spk1")
    # print(i)
    # print(ratioFemale)
    # print(ratioMale)
    if(is.na(ratioFemale) == FALSE & is.na(ratioMale) == FALSE) {
      if(ratioFemale > ratioMale) {
        genderCST <- "female"
      } else if(ratioMale > ratioFemale) {
        genderCST <- "male"
      } else {
        genderCST <- "female"
      }
    } else {
      genderCST <- "female"
    }
    
    #count number of female
    subset_care <- subset(subset, spker == "speaker_2")
    numFemale <- nrow(subset(subset_care, GLM == "female")) + nrow(subset(subset_care, RandomForest == "female")) + nrow(subset(subset_care, Xgboost == "female")) + nrow(subset(subset_care, GBM == "female"))
    ratioFemale_1 <- numFemale / nrow(subset_care) / 4
    #count number of male
    numMale <- nrow(subset(subset_care, GLM == "male")) + nrow(subset(subset_care, RandomForest == "male")) + nrow(subset(subset_care, Xgboost == "male")) + nrow(subset(subset_care, GBM == "male"))
    ratioMale_1 <- numMale / nrow(subset_care) / 4
    # print("spk2")
    # print(i)
    # print(ratioFemale_1)
    # print(ratioMale_1)

    if(is.na(ratioFemale_1) == FALSE & is.na(ratioMale_1) == FALSE )
    {
      if(ratioFemale_1 > ratioMale_1) {
        genderCare <- "female"
      } else if(ratioMale_1 > ratioFemale_1) {
        genderCare <- "male"
      } else {
        genderCare <- "female"
      }
    }
    else {
      genderCare <- "female"
    }
    x <- c(cst_1, ratioMale, ratioFemale, ratioMale_1, ratioFemale_1, genderCST, genderCare)
    x <- data.frame(x)
    rownames(x) <- c("subscriber", "spk1_pred_male", "spk1_pred_female", "spk2_pred_male", "spk2_pred_female", "gender_spk1", "gender_spk2")
    final_output <- rbind(final_output, t(x))
  }


  rownames(final_output) <- c(1:nrow(final_output))
  
  return(final_output)
}

########################################################################################################
## Lets run the code

if(extractFromSound == TRUE) {
  print("Converting audio to WAV ...")
  sound_files <- convertToWav(sound.files, mp3_dir = mp3_dir, wav_dir = wav_dir)
  print("Extracting Speech Diarization from audio...")
  spk <- getSpeechDiarization(sound_files)
  print("Extracting acoustic parameters from audio ...")
  file.remove("acoustic_parameters.csv")
  output <- extractAcoustic(spk, bp = c(0,22), wl = 2048, threshold = 5)
  
} else {
  output <- read.csv(file = "acoustic_parameters.csv")[,-1]
 
}

output[is.na(output)] <- 0

print("Predicting gender ...")
output_results <- predictGender(output)
print("Wrapping up ...")
final_output <- summarise(output_results)

write.csv(final_output, file = output_filename)
