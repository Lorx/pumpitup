# get data
source('C:/Users/Lewis/Desktop/datasciencechallenge/pumpitup/code/import_data.R')

#### EDA ####

### INIT ####
library(janitor)
library(tidyverse)
library(scales)
library(glue)
library(SmartEDA)
library(questionr)


#### Cleaning names ####

TestSetValues <- clean_names(TestSetValues)
TrainingSetLabels <- clean_names(TrainingSetLabels)
TrainingSetValues <- clean_names(TrainingSetValues)

#### Missing Values ####

sum(is.na(TrainingSetLabels)) 

# there is no missing values in training response

# Training data set

TrainingSet <- TrainingSetValues %>% left_join(TrainingSetLabels, by = "id")

glue('There is {percent(mean(is.na(TrainingSet)))} in Training set')

missingvalue <- purrr::map_df(TrainingSet, 
                    ~mean(is.na(.x))) %>% 
  pivot_longer(
    cols = names(.),
    names_to = "variable",
    values_to = "missing_value"
  ) %>% 
  arrange(desc(missing_value)) %>% 
  filter(missing_value > 0) %>% 
  mutate(missing_value = percent(missing_value))

glue('There are  {length(unique(missingvalue$variable))} that contain missing values :')

glue("{unique(missingvalue$variable)}")

# 47% of missing value in scheme_name. this variable will be dropped.

# The remaining will be treat according to their importance with the other features.


#### selecting data ####

train <- TrainingSet %>% select(-c(date_recorded, installer, num_private,
                          region_code, district_code, lga, recorded_by,
                          construction_year, extraction_type, extraction_type_group,
                          management_group, payment_type, water_quality, quantity,
                          waterpoint_type_group, longitude, latitude)) %>% 
  mutate_if(is.character, as.factor)

test <- TestSetValues %>% select(-c(date_recorded, installer, num_private,
                                    region_code, district_code, lga, recorded_by,
                                    construction_year, extraction_type, extraction_type_group,
                                    management_group, payment_type, water_quality, quantity,
                                    waterpoint_type_group, longitude, latitude)) %>% 
  mutate_if(is.character, as.factor)


#### Predicting power ####

# ExpReport(TrainingSet,Target="status_group",
#           label=NULL,op_file="test.html",
#           op_dir=getwd(),sc=2,sn=2,Rc="Yes")


predict_power <-
  ExpCatStat(
    train,
    Target = "status_group",
    result = "Stat",
    clim = 100,
    nlim = 500,
    Pclass = "Yes"
  )


#### v de cramer ####

df <- train

y <- "status_group"

x <- setdiff(names(df), y)

df2 <- df[,x]




cramer  <- matrix(NA,ncol(df2),4)

for (i in (1:ncol(df2)))
  
{     cramer[i,1] <- names(df2[i])

cramer[i,2] <- cramer.v(table(unlist(df2[,i]),unlist(df$status_group)))

cramer[i,3] <- chisq.test(table(unlist(df2[,i]),unlist(df$status_group)))$statistic

cramer[i,4] <- chisq.test(table(unlist(df2[,i]),unlist(df$status_group)))$p.value


}

colnames(cramer) <- c("variable","V de Cramer", "Fisher Stat", "p-value chi2")