
# Pseudocode RNN ----------------------------------------------------------
state_t <- 0
for (input_t in input_sequence) {
  output_t <- f(input_t, state_t)
  state_t <- output_t
}


# More detailed pseudocode for the RNN ------------------------------------

state_t <- 0
for (input_t in input_sequences) {
  output_t <- activation(dot(W, input_t) + dot(U, state_t) + b)
  state_t <- output_t
}


# Simple RNN implementation -----------------------------------------------

timesteps <- 100
input_features <- 32
output_features <- 64

random_array <- function(dim) {
  array(runif(prod(dim)), dim = dim)
}

inputs <- random_array(dim = c(timesteps, input_features))
state_t <- rep_len(0, length.out = c(output_features))

W <- random_array(dim = c(output_features, input_features))
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))

output_sequence <- array(0, dim = c(timesteps, output_features))
for (i in 1:nrow(inputs)) {
  input_t <- inputs[i, ]
  output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
  output_sequence[i, ] <- as.numeric(output_t)
  state_t <- output_t
}
