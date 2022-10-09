library("optparse")
library(stringr)

option_list <- list(
  make_option(c("-i", "--input_file"), type="character", default="default",
              help="cell file path", metavar="character"),
    make_option(c("-f", "--folds"), type="integer", default=10,
              help="numero di folds per il k-fold cv (se applicabile)", metavar="integer"),
   make_option(c("-t", "--to_train"), type="character", default="All",
              help="which model to train.
              Options:
              \n\tAll (Default)
              \n\tlr (Regressione lineare semplice)
              \n\tlrr (Regressione lineare con parametri ridotti)
              \n\tlrL1 (Regressione lineare con Lasso)
              \n\tlrL2 (Regressione lineare con Ridge)", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list);
opt <- parse_args(opt_parser);



if (opt$input_file == "default"){
  print_help(opt_parser)
  stop("input file argument is not present or wrong", call.=FALSE)
}

# constants
CELL_FILE <- opt$input_file
K_FOLDS <- opt$folds
METHOD <- opt$to_train

# paste(CELL_FILE)

a <- unlist(str_split(CELL_FILE, "/",))
CELL_NUMBER <- a[length(a)]
rm(a)

paste("Cella: ", CELL_NUMBER)
library(readr)



#leggiamo il dataset della cella
cella <- read_csv(CELL_FILE)
cella$Date <- NULL  # rimuove la prima colonna dato che ha l'orario

# TODO: continuare