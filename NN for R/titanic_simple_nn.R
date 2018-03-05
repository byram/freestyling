library(keras)
library(dplyr)
library(tidyr)
library(data.table)
set.seed(42)

#batch_size <- 128
#num_classes <- 2
#epochs <- 10

titanicData <- Titanic %>% 
  data.table(stringsAsFactors = T) %>% 
  mutate(N = as.integer(N)) %>% 
  {.[rep(1:nrow(.), .[, 5]), -5]} %>% 
  {.[sample(1:nrow(.)), ]}

titanicDataBin <- titanicData %>% 
  {cbind(to_categorical(.$Class)[, -1], 
         to_categorical(.$Sex)[, -1], 
         to_categorical(.$Age)[, -1], 
         to_categorical(.$Survived)[, -1])}

# Training dataset
partitionTrain <- titanicDataBin[1:round(nrow(titanicDataBin) * 0.8, 0), ]
xTrain <- partitionTrain[, -c(9, 10)]
yTrain <- partitionTrain[, c(9:10)]
# Testing dataset
xTest <- titanicDataBin[-c(1:nrow(partitionTrain)), -c(9, 10)]
yTest <- titanicDataBin[-c(1:nrow(partitionTrain)), c(9, 10)]

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 128, activation = "relu", input_shape = c(8)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 2, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer_adam(), 
  metrics = c("accuracy")
)

history <- model %>% fit(
  xTrain, yTrain, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

model %>% evaluate(xTest, yTest)
