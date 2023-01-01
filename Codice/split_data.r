
# The goal of this file is to split the original data in a single csv file for each cell.
# It is assumed that in the folder "root/Dati originali" there are csv files containing informations
# of all the cells within a certain period. The goal of this script is to combine all informations
# and aggregate them in a unique csv file containing information for each cell; splitted data are
# stored in the folder "root/CSV/Original data" under files named "Cella_XX.csv", where XX is the 
# number of the cell. Data are already sorted by ascending date
#
# This file is based on the script "read-data.rmd" originally written by Maurizio De Marchi
#
# Author: Matteo Dalle Vedove (matteodv99tn@gmail.com)

library(readr)
library(tidyverse)
library(lubridate)
library(dplyr)


#  _                    _   ____        _        
# | |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
# | |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
# | |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
# |_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                                

month_data      <- list()
file_pattern    <- '*csv$'  # must end with .csv

# Load all csv and append them in the month_data list
for(file in list.files(path = './Dati originali', pattern = file_pattern, full.names = TRUE)) {
    month_data <- append(month_data, list(read_csv2(file = file)))
}


#  ____        _ _ _     ____        _        
# / ___| _ __ | (_) |_  |  _ \  __ _| |_ __ _ 
# \___ \| '_ \| | | __| | | | |/ _` | __/ _` |
#  ___) | |_) | | | |_  | |_| | (_| | || (_| |
# |____/| .__/|_|_|\__| |____/ \__,_|\__\__,_|
#       |_|                                   

max_cell_count  <- 40
cells           <- unlist(list(rep(list(0), max_cell_count)), recursive = FALSE)

for(csv_data in month_data){ # for each csv file

    csv_data <- as_tibble(csv_data)
    csv_data <- csv_data[, -ncol(csv_data)]                 # remove last column that are just NAN
    csv_data <- csv_data %>% select(-contains('Tassullo'))  # remove unrelated data
    csv_data <- csv_data %>% select(-contains('Tuenetto'))  # remove unrelated data
    csv_data <- csv_data %>% select(-contains('Denno'))     # remove unrelated data
    csv_data <- csv_data %>% select(-contains('Cles'))      # remove unrelated data
    csv_data <- csv_data %>% select(-contains('Macchine'))  # remove unrelated data

    # remove common prefix "IpogeoX_"
    names(csv_data) <- gsub(pattern = 'Ipogeo.*_', replacement = '', x = names(csv_data))

    # extract columns that are shared for all cells (in particular we are intested in keeping the 
    # date-time column)
    common_predictors <- csv_data %>% select(-matches('Cella'))
    temp <- common_predictors
    common_predictors$Date <- as.POSIXct(common_predictors$Date, format = '%d-%m-%Y %H:%M:%S')
    
    for(i in 1:max_cell_count){

        # extract information of the "i"-th cell of interest (based on the prefix)
        single_cell <- csv_data %>% select(matches(paste0('Cella', i, '[A-Z]')))

        if (ncol(single_cell) == 0) {   # if there are no data
            next                        # it means that "i"-th cell do not exists; skip
        } else {
            names(single_cell) <- gsub( pattern = paste0('Cella', i),   # remove prefix
                                        replacement = '',  
                                        x = names(single_cell) )
            single_cell <- cbind(common_predictors, single_cell)        # to restore date-time col.

            df <- as.data.frame(cells[i])       # take already extracted data
            if(length(df) == 1){                # if it was not present the cell data, initialize it
                df <- single_cell
            } else {                            # otherwise append data of the current csv

                # Select only intersecting columns between already present data and previous data
                common_columns <- intersect(colnames(df), colnames(single_cell))
                df             <- rbind( subset(df,          select = common_columns), 
                                         subset(single_cell, select = common_columns))    
            }
            cells[i] <- list(df)                # re-insert the new data in the complete list

        }
    }
}


#   ____                          _     _           ____ ______     __
#  / ___|___  _ ____   _____ _ __| |_  | |_ ___    / ___/ ___\ \   / /
# | |   / _ \| '_ \ \ / / _ \ '__| __| | __/ _ \  | |   \___ \\ \ / / 
# | |__| (_) | | | \ V /  __/ |  | |_  | || (_) | | |___ ___) |\ V /  
#  \____\___/|_| |_|\_/ \___|_|   \__|  \__\___/   \____|____/  \_/   
                                                                     
dest_path <- paste0('CSV/Original data')
if( !dir.exists(dest_path)) {
    dir.create(path = dest_path)
} else {
    files <- list.files(dest_path, full.names = TRUE)
    unlink(files)
}

cell_index <- 0
for(cell in cells){
    
    cell_index <- cell_index + 1

    if(typeof(cell) == 'double')
        next

    # Due to some issues, data might have been stored in the original csv files with comma as 
    # decimal; to avoid this issue, we explicitly replate commas with dots and then convert to 
    # numeric. This will simplify operation in python.
    cell$TemperaturaCelle           <- gsub(pattern = ',', replacement = '.', fixed = TRUE, 
                                            x = cell$TemperaturaCelle)
    cell$TemperaturaMandataGlicole  <- gsub(pattern = ',', replacement = '.', fixed = TRUE, 
                                            x = cell$TemperaturaMandataGlicole)
    cell$TemperaturaRitornoGlicole  <- gsub(pattern = ',', replacement = '.', fixed = TRUE, 
                                            x = cell$TemperaturaRitornoGlicole)
    cell$TemperaturaCelle           <- as.numeric(cell$TemperaturaCelle         )  
    cell$TemperaturaMandataGlicole  <- as.numeric(cell$TemperaturaMandataGlicole)
    cell$TemperaturaRitornoGlicole  <- as.numeric(cell$TemperaturaRitornoGlicole)

    df <- as.data.frame(cell)
    df <- df[order(df$Date),]
  
    filename <- paste0(dest_path, '/Cella_', cell_index, '.csv')
    cat(paste('Saving', filename, '\n'))
    write.table(x = df, file = filename, col.names = TRUE, row.names = FALSE, sep = ',')

}
cat('Done!\n')