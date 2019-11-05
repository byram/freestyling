library(keras)

# Data import
max_features <- 10000
maxlen <- 20

imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

# Keras model fitting
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 8, 
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy", 
  metrics = c("acc")
)

summary(model)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, 
  batch_size = 32, 
  validation_split = 0.2
)
