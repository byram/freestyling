library(keras)

original_dataset_dir <- "D:/Downloads/train"

base_dir <- "D:/Downloads/cats_and_dogs_small"
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)
train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)
validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)
test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_cats_dir))
fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(validation_cats_dir))
fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(test_cats_dir))
fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_dogs_dir))
fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(validation_dogs_dir))
fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(test_dogs_dir))

### Data processing
datagen <- image_data_generator(
  rescale = 1 / 255, 
  rotation_range = 40, 
  width_shift_range = 0.2, 
  height_shift_range = 0.2, 
  shear_range = 0.2, 
  zoom_range = 0.2, 
  horizontal_flip = TRUE
)

# Example of data augmentation
fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[3]]
img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))
augmentation_generator <- flow_images_from_data(
  img_array, 
  generator = datagen, 
  batch_size = 1
)
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1, , , ]))
}
par(op)

# Use augmenter to adjust incoming training/validation data
#train_datagen <- image_data_generator(rescale = 1 / 255)
#validation_datagen <- image_data_generator(rescale = 1 / 255)
test_datagen <- image_data_generator(rescale = 1 / 255)
train_generator <- flow_images_from_directory(
  train_dir, 
  datagen, 
  target_size = c(150, 150), 
  batch_size = 32, 
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir, 
  test_datagen, 
  target_size = c(150, 150), 
  batch_size = 32, 
  class_mode = "binary"
)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", 
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  loss = "binary_crossentropy", 
  optimizer = optimizer_rmsprop(lr = 1e-4), 
  metrics = c("acc")
)

history <- model %>% fit_generator(
  train_generator, 
  steps_per_epoch = 100, 
  epochs = 100, 
  validation_data = validation_generator, 
  validation_steps = 50
)
model %>% save_model_hdf5("cats_and_dogs_small_2.h5")
plot(history)
