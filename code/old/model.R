### CHARGEMENT DES PACKAGES ####

source('C:/Users/Lewis/Desktop/datasciencechallenge/pumpitup/code/eda.R')

library(h2o)

# launching H20 with a maximum of 4 gb Ram

localH2O <<- h2o.init(ip =  "localhost", port = 54321, 
                      nthreads = -1, max_mem_size = "10g")


#### TRAINING SET ####
y <- "status_group"

x <- setdiff(names(df), y)


BM <- h2o.splitFrame(as.h2o(df), ratios = 0.75)

BM_train <- BM[[1]]
BM_test <- BM[[2]]

#### SUBMITION SET ####

submit.hex <- as.h2o(TestSetValues,
                   
                   destination_frame= "submition_data.hex")



#### AUTOML TRY ####

h2o.flow()
