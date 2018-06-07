library(keras)
library(dplyr)
library(data.table)
set.seed(42)

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
partitionTrain <- titanicDataBin[1:round(nrow(titanicDataBin) * 0.75, 0), ]
xTrain <- partitionTrain[, -c(9, 10)]
yTrain <- partitionTrain[, c(9:10)]
# Testing dataset
xTest <- titanicDataBin[-c(1:nrow(partitionTrain)), -c(9, 10)]
yTest <- titanicDataBin[-c(1:nrow(partitionTrain)), c(9, 10)]
xLogTest <- titanicData[-c(1:nrow(partitionTrain)), -4]
yLogTest <- titanicData[-c(1:nrow(partitionTrain)), 4]

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 6, activation = "relu", input_shape = c(8)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 2, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer_adam(), 
  metrics = c("accuracy")
)

history <- model %>% fit(
  xTrain, yTrain, 
  epochs = 20, batch_size = 4, 
  validation_split = 1 / 3
)

model %>% 
  evaluate(xTest, yTest) %>% 
  {cat("Neural Network accuracy: ", round(.[[2]] * 100, 1), "%", sep = "")}

logreg <- glm(Survived ~ Class + Age + Sex, family = binomial, data = titanicData) %>% 
  {factor(predict(., newdata = xLogTest) > 0, 
          labels = c("Survived", "Died"))} %>% 
  as.numeric() %>% 
  {sum(. == as.numeric(yLogTest)) / length(yLogTest)} %>% 
  {cat("Logistic Regression accuracy: ", round(. * 100, 1), "%", sep = "")}
