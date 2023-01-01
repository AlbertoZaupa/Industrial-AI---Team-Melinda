# This file preprocess all csv files contained in the folder "root/CSV/Original data" by applying 
# a moving average (aka rolling mean) filter on some variables. The results are stored in the folder 
# "root/CSV/Processed data". 
#
# author: Matteo Dalle Vedove (matteodv99tn@gmail.com)

library(readr)
library(tidyverse)
library(tidyr)
library(zoo)

# get absolute path of the current script and then generate the root and csv paths
SCRIPT_PATH <- normalizePath(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "."))
ROOT_PATH   <- normalizePath(file.path(SCRIPT_PATH, "..", ".."))
CSV_PATH    <- file.path(ROOT_PATH, "CSV", "Original data")
OUT_PATH    <- file.path(ROOT_PATH, "CSV", "Processed data")

# if the directory for the processed csv data is not present, create it
if(!dir.exists(OUT_PATH)) dir.create(OUT_PATH)

# load all csv files in the CSV folder
csv_files <- list.files(path = CSV_PATH, pattern = '*csv$', full.names = FALSE)


#  ____                              _             
# |  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _ 
# | |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
# |  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |
# |_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                                            |___/ 

process_csv_file <- function(file_name) {

    cat('Processing file:', file_name, '\n')

    # load the csv file
    csv_data <- read_csv(file = file.path(CSV_PATH, file_name), show_col_types = FALSE)
    csv_data <- csv_data %>% drop_na()

    # data processing 
    pump_on = csv_data$PompaGlicoleMarcia == 1

    modifiable_columns <- c('TemperaturaCelle', 
                            'TemperaturaMandataGlicole', 
                            'TemperaturaRitornoGlicole',
                            'PercentualeAperturaValvolaMiscelatrice')

    for(var in modifiable_columns) {
        csv_data[[var]][pump_on]  <- rollapply( csv_data[[var]][pump_on], width = 50, 
                                                FUN = mean, align = 'center', fill = NA)
        csv_data[[var]][!pump_on] <- rollapply( csv_data[[var]][!pump_on], width = 50, 
                                                FUN = mean, align = 'center', fill = NA)
    }

    csv_data$TemperaturaCelle[pump_on] <- rollapply(csv_data$TemperaturaCelle[csv_data$PompaGlicoleMarcia == 1], width = 50, FUN = mean, align = 'center', fill = NA)

    csv_data$TemperaturaCelle 

    # write back csv file
    write.table(csv_data, file = file.path(OUT_PATH, file_name), 
        col.names = TRUE, row.names = FALSE, sep = ',')    
}


#  ____  _       _       
# |  _ \| | ___ | |_ ___ 
# | |_) | |/ _ \| __/ __|
# |  __/| | (_) | |_\__ \
# |_|   |_|\___/ \__|___/

plot_cell_temperatures <- function(data, begin = NULL, end = NULL){

    if(is.null(begin)) begin <- min(data$Date)    
    if(is.null(end))   end   <- as.Date(begin) + 5

    data <- data[data$Date >= begin & data$Date <= end, ]   

    plot(data$Date, data$TemperaturaMandataGlicole, 
        type = 'l', col = 'green', xlab = 'Date', ylab = 'Temperature (°C)')
    lines(data$Date, data$TemperaturaRitornoGlicole, col = 'Orange')
    lines(data$Date, data$TemperaturaCelle, col = 'red')
    lines(data$Date, data$TemperaturaMandataGlicoleNominale, col = 'grey')

    grid()
    legend('topright', 
        legend = c( 'Mandata Glicole', 
                    'Ritorno Glicole', 
                    'Cella', 
                    'Mandata Glicole Nominale'), 
        col = c('green', 'Orange', 'red', 'grey'), lty = 1, cex = 0.8 )
}


#  __  __       _       
# |  \/  | __ _(_)_ __  
# | |\/| |/ _` | | '_ \ 
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|
 
for(file in csv_files) process_csv_file(file)

original_data <- read_csv(file = file.path(CSV_PATH, csv_files[1]), show_col_types = FALSE)
modified_data <- read_csv(file = file.path(OUT_PATH, csv_files[1]), show_col_types = FALSE)

par(mfrow = c(1,2))
plot_cell_temperatures(original_data)
plot_cell_temperatures(modified_data)