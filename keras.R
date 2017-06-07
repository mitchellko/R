library(tensorflow)
use_condaenv("tf")

install.packages("kerasR")
library(kerasR)

model <- keras_model_sequential() 
