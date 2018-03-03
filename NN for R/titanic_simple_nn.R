require(keras)
require(dplyr)
require(tidyr)
require(data.table)
set.seed(42)

batch_size <- 128
num_classes <- 2
epochs <- 10

titanicData <- Titanic %>%
  data.table() %>%
  mutate(N = as.integer(N)) %>%
  {.[rep(1:nrow(.), .[, 5]), -5]} %>%
  {.[sample(1:nrow(.)), ]}

# Training dataset
xTrain <- titanicData[1:round(nrow(titanicData) * 0.8, 0), -4]

yTrain <- titanicData[1:nrow(xTrain), 4] %>%
  mutate()


xTrain <- titanicData[1:round(nrow(titanicData), 0), -4]
yTrain <- titanicData[1:round(nrow(titanicData), 0), 4]