library(keras)

mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y

x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Model 1, RMSProp optimizer
model <- keras_model_sequential() %>%  
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer_rmsprop(), 
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 15, batch_size = 128, 
  validation_split = 0.2
)
plot(history)

model %>% evaluate(x_test, y_test) %>% print()

#model %>% predict_classes(x_test)

# Model 2, Adam optimizer
model2 <- keras_model_sequential()
model2 %>% 
  layer_dense(units = 256, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, activation = "softmax")

model2 %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer_adam(), 
  metrics = c("accuracy")#, 
  #config = tf$device('/cpu:0') # this line should force TensorFlow to run on CPU
)

history2 <- model2 %>% fit(
  x_train, y_train, 
  epochs = 15, batch_size = 128, 
  validation_split = 0.2
)
plot(history2)

model2 %>% evaluate(x_test, y_test) %>% print()

#model2 %>% predict_classes(x_test)
