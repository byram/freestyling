
#### Simple one-hot encoder for words ####
samples <- c("The cat sat on the mat.", "The dog ate my homework.")

token_index <- list()
for (sample in samples) {
  for (word in strsplit(sample, " ")[[1]]) {
    if (!word %in% names(token_index)) {
      token_index[[word]] <- length(token_index) + 2
    }
  }
}

max_length <- 10

results <- array(0, dim = c(length(samples), 
                            max_length, 
                            max(as.integer(token_index))))

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    index <- token_index[[words[[j]]]]
    results[[i, j, index]] <- 1
  }
}

#### Using Keras to one-hot encode text ####
library(keras)

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

tokenizer <- text_tokenizer(num_words = 1000) %>% 
  fit_text_tokenizer(samples)

sequences <- texts_to_sequences(tokenizer, samples)

one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")

# Alternatively, you can directly get the one-hot binary representations
word_index <- tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")
