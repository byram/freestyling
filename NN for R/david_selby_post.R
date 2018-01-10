# This code and the notes are taken from or derived from the post found here: http://selbydavid.com/2018/01/09/neural-network/

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

train <- function(x, y, hidden, learnRate, iterations) {
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

