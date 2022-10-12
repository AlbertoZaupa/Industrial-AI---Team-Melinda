library("optparse")
library(stringr)

option_list <- list(
  make_option(c("-i", "--input-file"), type = "character", default = "default",
              help = "path del file della cella", metavar = "character"),
  make_option(c("-o", "--output-dir"), type = "character", default = "./",
              help = "directory dove salvare il modello", metavar = "character"),
  make_option(c("-m", "--metrics-file"), type = "character", default = "./metrics.csv",
              help = "file dove scrivere le metriche del modello", metavar = "character"),
  make_option(c("-y", "--variable-to-fit"), type = "character", default = "TemperaturaCelle",
              help = "Variabile da fittare", metavar = "character"),
  make_option(c("-p", "--p-value"), type = "double", default = 1e-2,
              help = "p-value minimo per includere i predittori", metavar = "double"),
  make_option(c("-c", "--cella"), type = "integer", default = -1,
              help = "numero cella", metavar = "integer"),
  make_option(c("-t", "--test-percentage"), type = "double", default = 0.2,
              help = "percentuale da dedicate al testing", metavar = "double"),
  make_option(c("-e", "--error-metric"), type = "character", default = 'mse',
              help = "metrica di errore da stampare", metavar = "character"),
  make_option(c("-v", "--verbose"), type = "logical", default = FALSE,
              help = "Se trascrivere tutti i passaggi", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list);
opt <- parse_args(opt_parser);


if (opt$`input-file` == "default") {
  print_help(opt_parser)
  stop("file di input fornito errato o assente", call. = FALSE)
}
if (opt$cella < 0) {
  print_help(opt_parser)
  stop("Fornire il numero della cella (assente o minore di 0)", call. = FALSE)
}

# constants
CELL_FILE <- opt$`input-file`
K_FOLDS <- opt$folds
P_VALUE <- opt$`p-value`
OUTPUT_DIR <- opt$`output-dir`
VERBOSE <- opt$verbose
CELLA <- opt$cella
TEST_SIZE <- opt$`test-percentage`
y <- opt$`variable-to-fit`
VERBOSE_LOG <- ""
METRIC <- opt$`error-metric`
METRIC_FILE <- opt$`metrics-file`

log_info <- function(verbose_string, new_string) {
  verbose_string <- paste0(verbose_string, "\n", new_string)
  print(new_string)
  return(verbose_string)
}

rm(opt, option_list, opt_parser)

if (VERBOSE) {
  VERBOSE_LOG <- log_info(VERBOSE_LOG,
                          paste("Input file:", CELL_FILE))
  VERBOSE_LOG <- log_info(VERBOSE_LOG,
                          paste("Cella:", CELLA))
  VERBOSE_LOG <- log_info(VERBOSE_LOG,
                          paste("Folds per k-fold CV:", K_FOLDS))
  VERBOSE_LOG <- log_info(VERBOSE_LOG,
                          paste("Directory per l'output:", OUTPUT_DIR))
  VERBOSE_LOG <- log_info(VERBOSE_LOG,
                          paste("P-value massimo:", P_VALUE))
}

if (VERBOSE) {
  print("Loading library: readr")
}
library(readr)

#leggiamo il dataset della cella
cella <- read_csv(CELL_FILE, show_col_types = VERBOSE)
cella$Date <- NULL  # rimuove la prima colonna dato che ha l'orario
cella$TemperaturaMandataGlicoleNominale <- NULL

if (VERBOSE) {
  library(tidyverse)

  cella <- as_tibble(cella)
  summary(cella) # assicuriamoci che vada tutto# bene
}

if (VERBOSE) {
  print("Loading library Metrics")
}
library(Metrics)

# suddivisione
DATASET_SIZE <- nrow(cella)
# TUNING_SIZE <- 0.8

id_test <- as.integer(DATASET_SIZE * (1 - TEST_SIZE)):DATASET_SIZE

cella_test <- cella[id_test,]
cella_train <- cella[-id_test,]


compute_metrics <- function(name = "dummy", preds = c(2, 1, 3, 8, 8, 2, 3, 6), test = c(7, 3, 4, 5, 6, 8, 5, 6)) {
  error <- data.frame(name = name,
                      mse = mse(actual = test, predicted = preds), # mean square error
                      rmse = rmse(actual = test, predicted = preds), # root mean square error
                      rmsle = rmsle(actual = test, predicted = preds), # root mean square log error
                      rrse = rrse(actual = test, predicted = preds), # root relative square error
                      rse = rse(actual = test, predicted = preds), # relative square error
                      bias = bias(actual = test, predicted = preds), # bias del modello
                      pbias = percent_bias(actual = test, predicted = preds), # percent bias
                      mae = mae(actual = test, predicted = preds), # Mean Absolute Error
                      rae = rae(actual = test, predicted = preds), # relative absolute error
                      mape = mape(actual = test, predicted = preds), # Mean Absolute Percentage Error
                      mdae = mdae(actual = test, predicted = preds), # Median Absolute Different Error
                      msle = msle(actual = test, predicted = preds), # Mean Squared Log Error
                      smape = smape(actual = test, predicted = preds), #symmetric mean absolute percentage error
                      sse = sse(actual = test, predicted = preds), # sum of the squared differences between two numeric vectors
                      r2 = cor(test, preds)^2
  )

  return(error)
}

if (VERBOSE) {
  print("Loading library stringr")
}
library(stringr)

create_formula <- function(y = "y", useful_factors = ".") {
  result <- paste(y, "~ ")
  for (factor in useful_factors) {
    result <- paste(result, factor, "+")
  }

  result <- substring(result, first = 1, str_length(result) - 1)
  return(result)
}

if (VERBOSE) {
  VERBOSE_LOG <- log_info(VERBOSE_LOG, "Staring parameter selection")
}

library(glmnet)
library(Metrics)

formule <- paste0(y, " ~ .")

if(VERBOSE){
  print(paste0("First model: ", formule))
}

while (TRUE) {
  model <- glm(formula = formule, family = gaussian, data = cella_train)
  n <- length(model$coefficients)-1
  s <- summary(model)
  useful_predictors <- names(which(s$coefficients[, "Pr(>|t|)"] < P_VALUE)[-1]) # -1 per l-intercept
  if(length(useful_predictors) == n){break}
  formule <- create_formula(y, useful_predictors)
  if(VERBOSE){
    print(paste0("New formula: ", formule))
    VERBOSE_LOG <- log_info(VERBOSE_LOG,paste0("New formula: ", formule))
  }

}

if(VERBOSE){
  print(s)
}

predictions <- predict(model, newdata = cella_test)
errors <- compute_metrics(name = "Linear Regression",preds = predictions, test = unlist(cella_test[y]))
write_csv(errors, paste0(OUTPUT_DIR, METRIC_FILE))
print(errors[METRIC])
if(VERBOSE){
  VERBOSE_LOG <- log_info(VERBOSE_LOG, paste0(METRIC,": ",errors[METRIC]))
  print("training and saving final model")
}

model <- glm(formula = formule, family = gaussian, data = cella)
saveRDS(model, file = paste0(OUTPUT_DIR,"modello_cella_", CELLA,".rds"))

if(VERBOSE){
  log_info(VERBOSE_LOG, paste0("Model saved as: ",OUTPUT_DIR,"modello_cella_", CELLA,".rds"))

  fileConn<-file(paste0(OUTPUT_DIR, "log.txt"))
  writeLines(VERBOSE_LOG, fileConn)
  close(fileConn)
}