library(keras)
library(EBImage)

# Import data
directory <- "data/CNN_example/"
pic1 <- c('p1.jpg', 'p2.jpg', 'p3.jpg', 'p4.jpg', 'p5.jpg', 
          'c1.jpg', 'c2.jpg', 'c3.jpg', 'c4.jpg', 'c5.jpg', 
          'b1.jpg', 'b2.jpg', 'b3.jpg', 'b4.jpg', 'b5.jpg')
train <- list()
for (i in 1:15) {
  train[[i]] <- readImage(paste(directory, pic1[i], sep = ""))
}

pic2 <- c('p6.jpg', 'c6.jpg', 'b6.jpg')
test <- list()
for (i in 1:3) {
  test[[i]] <- readImage(paste(directory, pic2[i], sep = ""))
}

# Explore
train[[12]]
summary(train[[12]])
display(train[[12]])
plot(train[[12]])

par(mfrow = c(3, 5))
for (i in 1:15) plot(train[[i]])
par(mfrow = c(1, 1))

# Resize & combine
str(train)
for (i in 1:15) {train[[i]] <- resize(train[[i]], 100, 100)}
for (i in 1:3) {test[[i]] <- resize(test[[i]], 100, 100)}

train <- combine(train)
x <- tile(train, 5)
display(x, title = "Pictures")

test <- combine(test)
y <- tile(test, 3)
display(y, title = "Pictures")

# Reorder dimensions
train <- aperm(train, c(4, 1, 2, 3))
test <- aperm(test, c(4, 1, 2, 3))

# Response
trainy <- c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
testy <- c(0, 1, 2)

# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# Model
model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3, 3), 
                activation = 'relu', 
                input_shape = c(100, 100, 3)) %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, 
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_conv_2d(filters = 64, 
                kernel_size = c(3, 3), 
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 3, activation = 'softmax') %>%
  compile(loss = 'categorical_crossentropy', 
          # Using Adam instead of SGD produced almost certain beliefs from the 
          # model, which were also 100% correct over this small dataset.
          optimizer = optimizer_adam(), 
          metrics = c('accuracy'))

# Fit model
history <- model %>%
  fit(train, 
      trainLabels, 
      epochs = 60, 
      batch_size = 32, 
      validation_split = 0.2)
#plot(history)

# Evaluation & Prediction - training data
model %>% evaluate(train, trainLabels)
pred <- model %>% predict_classes(train)
table(Predicted = pred, Actual = trainy)

prob <- model %>% predict_proba(train)
cbind(prob, predicted_class = pred, Actual = trainy)

# Evaluation & Prediction - testing data
model %>% evaluate(test, testLabels)
pred <- model %>% predict_classes(test)
table(Predicted = pred, Actual = testy)

prob <- model %>% predict_proba(test)
cbind(prob, Predicted_class = pred, Actual = testy)

