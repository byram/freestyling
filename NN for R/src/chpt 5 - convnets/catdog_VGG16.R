library(keras)

conv_base <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE, 
  input_shape = c(150, 150, 3)
)

base_dir <- "D:/Downloads/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

### First method: use convolutional base once, then feed that result into a 
### densely connected model.
datagen <- image_data_generator(rescale = 1 / 255)
batch_size <- 20

extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory, 
    generator = datagen, 
    target_size = c(150, 150), 
    batch_size = batch_size, 
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    features[index_range, , , ] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = c(4 * 4 * 512)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5), 
  loss = "binary_crossentropy", 
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels, 
  epochs = 30, 
  batch_size = 20, 
  validation_data = list(validation$features, validation$labels)
)

### Second method: map densely connected layer directly on top of the 
### convolutional base once and run entire model end to end.
model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

freeze_weights(conv_base)

train_datagen <- image_data_generator(
  rescale = 1 / 255, 
  rotation_range = 40, 
  width_shift_range = 0.2, 
  height_shift_range = 0.2, 
  shear_range = 0.2, 
  zoom_range = 0.2, 
  horizontal_flip = TRUE, 
  fill_mode = "nearest"
)
test_datagen <- image_data_generator(rescale = 1 / 255)
train_generator <- flow_images_from_directory(
  train_dir, 
  train_datagen, 
  target_size = c(150, 150), 
  batch_size = 20, 
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir, 
  test_datagen, 
  target_size = c(150, 150), 
  batch_size = 20, 
  class_mode = "binary"
)

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5), 
  loss = "binary_crossentropy", 
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator, 
  steps_per_epoch = 100, 
  epochs = 30, 
  validation_data = validation_generator, 
  validation_steps = 50
)


### Third method: fine-tuning the highest convolution along with our own final 
### densely connected layer.
unfreeze_weights(conv_base, from = "block3_conv1")

model %>% compile(
  loss = "binary_crossentropy", 
  optimizer = optimizer_rmsprop(lr = 1e-5), 
  metrics = c("acc")
)

history <- model %>% fit_generator(
  train_generator, 
  steps_per_epoch = 100, 
  epochs = 100, 
  validation_data = validation_generator, 
  validation_steps = 50
)

test_generator <- flow_images_from_directory(
  test_dir, 
  test_datagen, 
  target_size = c(150, 150), 
  batch_size = 20, 
  class_mode = "binary"
)
model %>% evaluate_generator(test_generator, steps = 50) %>% print()
