# This code and the notes are taken from or derived from the post found here: http://selbydavid.com/2018/01/09/neural-network/

two_spirals <- function(N = 200, 
                        radians = 3 * pi, 
                        theta0 = pi / 2, 
                        labels = 0:1) {
  N1 <- floor(N / 2)
  N2 <- N - N1
  theta <- theta0 + runif(N1) * radians
  spiral1 <- cbind(-theta * cos(theta) + runif(N1),
                   theta * sin(theta) + runif(N1))
  spiral2 <- cbind(theta * cos(theta) + runif(N2),
                   -theta * sin(theta) + runif(N2))
  points <- rbind(spiral1, spiral2)
  classes <- c(rep(0, N1), rep(1, N2))
  data.frame(x1 = points[, 1],
             x2 = points[, 2],
             class = factor(classes, labels = labels))
}
set.seed(42)
hotdogs <- two_spirals(labels = c('not hot dog', 'hot dog'))

library(ggplot2)
theme_set(theme_classic())
ggplot(hotdogs, aes(x = x1, y = x2, col = class)) + 
  geom_point()

logreg <- glm(class ~ x1 + x2, family = binomial, data = hotdogs)
correct <- sum((fitted(logreg) > .5) + 1 == as.integer(hotdogs$class))

beta <- coef(logreg)
grid <- expand.grid(x1 = seq(min(hotdogs$x1) - 1, 
                             max(hotdogs$x1) + 1, 
                             by = 0.25), 
                    x2 = seq(min(hotdogs$x2) - 1, 
                             max(hotdogs$x2) + 1, 
                             by = 0.25))
grid$class <- factor((predict(logreg, newdata = grid) > 0) * 1, 
                     labels = c("not hot dog", "hot dog"))
ggplot(hotdogs, aes(x1, x2, col = class)) + 
  geom_point(data = grid, size = 0.5) + 
  geom_point() + 
  labs(x = expression(x[1]), y = expression(x[2])) + 
  geom_abline(intercept = -beta[1] / beta[3], 
              slope = -beta[2] / beta[3])

sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

feedforward <- function(x, w1, w2) {
  z1 <- cbind(1, x) %*% w1
  h <- sigmoid(z1)
  z2 <- cbind(1, h) %*% w2
  list(output = sigmoid(z2), h = h)
}

backpropogate <- function(x, y, yHat, w1, w2, h, learnRate) {
  dw2 <- t(cbind(1, h)) %*% (yHat - y)
  dh  <- (yHat - y) %*% t(w2[-1, , drop = FALSE])
  dw1 <- t(cbind(1, x)) %*% (h * (1 - h) * dh)
  w1  <- w1 - learnRate * dw1
  w2  <- w2 - learnRate * dw2
  list(w1 = w1, w2 = w2)
}

train <- function(x, y, hidden = 5, learnRate = 1e-2, iterations = 1e4) {
  d  <- ncol(x) + 1
  w1 <- matrix(rnorm(d * hidden), d, hidden)
  w2 <- as.matrix(rnorm(hidden + 1))
  for (i in 1:iterations) {
    ff <- feedforward(x, w1, w2)
    bp <- backpropogate(x, y, yHat = ff$output, w1, w2, 
                        h = ff$h, learnRate = learnRate)
    w1 <- bp$w1; w2 <- bp$w2
  }
  list(output = ff$output, w1 = w1, w2 = w2)
}

x <- data.matrix(hotdogs[, c("x1", "x2")])
y <- hotdogs$class == "hot dog"

# Neural net with 5 hidden neurons
nnet5 <- train(x, y, hidden = 5, iterations = 1e5)
print(mean((nnet5$output > 0.5) == y))
ff_grid <- feedforward(x = data.matrix(grid[, c("x1", "x2")]), 
                       w1 = nnet5$w1, 
                       w2 = nnet5$w2)
grid$class <- factor((ff_grid$output > 0.5) * 1, 
                     labels = levels(hotdogs$class))
ggplot(hotdogs, aes(x1, x2, col = class)) + 
  geom_point(data = grid, size = 0.5) + 
  geom_point() + 
  labs(x = expression(x[1]), y = expression(x[2]))

# Neural net with 30 hidden neurons
nnet30 <- train(x, y, hidden = 30, iterations = 1e5)
print(mean((nnet30$output > 0.5) == y))
ff_grid <- feedforward(x = data.matrix(grid[, c("x1", "x2")]), 
                       w1 = nnet30$w1, 
                       w2 = nnet30$w2)
grid$class <- factor((ff_grid$output > 0.5) * 1, 
                     labels = levels(hotdogs$class))
ggplot(hotdogs, aes(x1, x2, col = class)) + 
  geom_point(data = grid, size = 0.5) + 
  geom_point() + 
  labs(x = expression(x[1]), y = expression(x[2]))

library(R6)
NeuralNetwork <- R6Class("NeuralNetwork", 
                         public = list(
                           X = NULL,  Y = NULL, 
                           W1 = NULL, W2 = NULL, 
                           output = NULL, 
                           initialize = function(formula, hidden, data = list()) {
                             # Model and training data
                             mod <- model.frame(formula, data = data)
                             self$X <- model.matrix(attr(mod, 'terms'), data = mod)
                             self$Y <- model.response(mod)
                             # Dimensions
                             D <- ncol(self$X) # input dimensions (+ bias)
                             K <- length(unique(self$Y)) # number of classes
                             H <- hidden # number of hidden nodes (- bias)
                             # Initial weights and bias
                             self$W1 <- .01 * matrix(rnorm(D * H), D, H)
                             self$W2 <- .01 * matrix(rnorm((H + 1) * K), H + 1, K)
                           }, 
                           fit = function(data = self$X) {
                             h <- self$sigmoid(data %*% self$W1)
                             score <- cbind(1, h) %*% self$W2
                             return(self$softmax(score))
                           }, 
                           feedforward = function(data = self$X) {
                             self$output <- self$fit(data)
                             invisible(self)
                           }, 
                           backpropagate = function(lr = 1e-2) {
                             h <- self$sigmoid(self$X %*% self$W1)
                             Yid <- match(self$Y, sort(unique(self$Y)))
                             haty_y <- self$output - (col(self$output) == Yid) # E[y] - y
                             dW2 <- t(cbind(1, h)) %*% haty_y
                             dh <- haty_y %*% t(self$W2[-1, , drop = FALSE])
                             dW1 <- t(self$X) %*% (self$dsigmoid(h) * dh)
                             self$W1 <- self$W1 - lr * dW1
                             self$W2 <- self$W2 - lr * dW2
                             invisible(self)
                           }, 
                           predict = function(data = self$X) {
                             probs <- self$fit(data)
                             preds <- apply(probs, 1, which.max)
                             levels(self$Y)[preds]
                           }, 
                           compute_loss = function(probs = self$output) {
                             Yid <- match(self$Y, sort(unique(self$Y)))
                             correct_logprobs <- -log(probs[cbind(seq_along(Yid), Yid)])
                             sum(correct_logprobs)
                           }, 
                           train = function(iterations = 1e4, 
                                            learn_rate = 1e-2, 
                                            tolerance = .01, 
                                            trace = 100) {
                             for (i in seq_len(iterations)) {
                               self$feedforward()$backpropagate(learn_rate)
                               if (trace > 0 && i %% trace == 0)
                                 message('Iteration ', i, '\tLoss ', self$compute_loss(), 
                                         '\tAccuracy ', self$accuracy())
                               if (self$compute_loss() < tolerance) break
                             }
                             invisible(self)
                           }, 
                           accuracy = function() {
                             predictions <- apply(self$output, 1, which.max)
                             predictions <- levels(self$Y)[predictions]
                             mean(predictions == self$Y)
                           }, 
                           sigmoid = function(x) 1 / (1 + exp(-x)), 
                           dsigmoid = function(x) x * (1 - x), 
                           softmax = function(x) exp(x) / rowSums(exp(x))
                         )
)

irisnet <- NeuralNetwork$new(Species ~ ., data = iris, hidden = 5)
irisnet$train(9999, trace = 1e3, learn_rate = 0.0001)
irisnet

