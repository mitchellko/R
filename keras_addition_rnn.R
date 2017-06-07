devtools::install_github("rstudio/keras")
library(keras)

library(stringi)

learn_encoding <- function(chars){
    sort(chars)
}

encode <- function(char, char_table){
    strsplit(char, "") %>%
        unlist() %>%
        sapply(function(x){
            as.numeric(x == char_table)
        }) %>% 
        t()
}

decode <- function(x, char_table){
    apply(x,1, function(y){
        char_table[which.max(y)]
    }) %>% paste0(collapse = "")
}

generate_data <- function(size, digits, invert = TRUE){
    
    max_num <- as.integer(paste0(rep(9, digits), collapse = ""))
    
    # generate integers for both sides of question
    x <- sample(1:max_num, size = size, replace = TRUE)
    y <- sample(1:max_num, size = size, replace = TRUE)
    
    # make left side always samalller then right side
    left_side <- ifelse(x <= y, x, y)
    right_side <- ifelse(x >= y, x, y)
    
    results <- left_side + right_side
    
    # pad with spaces on the right
    questions <- paste0(left_side, "+", right_side)
    questions <- stri_pad(questions, width = 2*digits+1, 
                          side = "right", pad = " ")
    if(invert){
        questions <- stri_reverse(questions)
    }
    # pad with spaces on the left
    results <- stri_pad(results, width = digits + 1, 
                        side = "left", pad = " ")
    
    list(
        questions = questions,
        results = results
    )
}

TRAINING_SIZE <- 50000
DIGITS <- 2

MAXLEN <- DIGITS + 1 + DIGITS

charset <- c(0:9, "+", " ")
char_table <- learn_encoding(charset)

examples <- generate_data(size = TRAINING_SIZE, digits = DIGITS)

x <- array(0, dim = c(length(examples$questions), MAXLEN, length(char_table)))
y <- array(0, dim = c(length(examples$questions), DIGITS + 1, length(char_table)))

for(i in 1:TRAINING_SIZE){
    x[i,,] <- encode(examples$questions[i], char_table)
    y[i,,] <- encode(examples$results[i], char_table)
}

# Shuffle
indices <- sample(1:TRAINING_SIZE, size = TRAINING_SIZE)
x <- x[indices,,]
y <- y[indices,,]

# Explicitly set apart 10% for validation data that we never train over.
split_at <- trunc(TRAINING_SIZE/10)
x_val <- x[1:split_at,,]
y_val <- y[1:split_at,,]
x_train <- x[(split_at + 1):TRAINING_SIZE,,]
y_train <- y[(split_at + 1):TRAINING_SIZE,,]

print('Training Data:')
print(dim(x_train))
print(dim(y_train))

print('Validation Data:')
print(dim(x_val))
print(dim(y_val))

# Training ----------------------------------------------------------------
HIDDEN_SIZE <- 128
BATCH_SIZE <- 128
LAYERS <- 1

# Initialize sequential model
model <- keras_model_sequential() 

model %>%
    layer_lstm(HIDDEN_SIZE, input_shape=c(MAXLEN, length(char_table))) %>%
    layer_repeat_vector(DIGITS + 1)

for(i in 1:LAYERS)
    model %>% layer_lstm(HIDDEN_SIZE, return_sequences = TRUE)

model %>% 
    time_distributed(layer_dense(units = length(char_table))) %>%
    layer_activation("softmax")

# Compiling the model
model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = "adam", 
    metrics = "accuracy"
)

# Get the model summary
summary(model)

# Fitting loop
model %>% fit( 
    x = x_train, 
    y = y_train, 
    batch_size = BATCH_SIZE, 
    epochs = 70,
    validation_data = list(x_val, y_val)
)

# Predict for a new obs
new_obs <- encode("55+22", char_table) %>%
    array(dim = c(1,5,12))
result <- predict(model, new_obs)
result <- result[1,,]
decode(result, char_table)

