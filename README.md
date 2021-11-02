# Stroke Prediction Model 
Using some clinical data gotten from Kaggle (https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) I created a model to predict the probability of developing a stroke, and examined the factors which are most predictive of developing one. 
This project was done with R to start. Over the next few weeks I'll be including a python script that carries out the same functions as the R script. 


## What is a Stroke? 
A stroke is a condition that occurs when blood supply to a part of the brain is cut off. This prevents oxygen from getting to brain tissue, and can cause tissue death within a few minutes. Strokes are considered a medical emergency, with treatment necessary for the condition. Serious strokes can lead to: 
- Paralysis/loss of muscle function 
- Difficulty talking or swallowing 
- Emotional problems 
- Death 

A stroke is a serious issue, and there are many risk factors which can be considered when assessing an individual's risk of a stroke. These assessments are generally made by physicians, but we can also use data to make an educated guess regarding an individual's risk of developing a stroke. 


## Step 1: Load and Explore the data 

This was done with the lines of code below: 

```
library(pacman)
p_load(tidyverse, # data manipulation/visualization
       caret, # machine learning package
       ggthemes, # additional themes
       mice, # assessing missing vars
       rpart, # decision trees
       rpart.plot, # plotting said trees
       effects # to visualize regression coefficient effects
       )

stroke <- read_csv("data/healthcare-dataset-stroke-data.csv) %>%
          select(-c(id)) # unnecessary column
str(stroke)
```
![image](https://user-images.githubusercontent.com/91495866/139961856-c269f650-832b-4648-9912-5609fe596266.png)

Next thing is to adjust the data to make sure everything is in order, and check for missing values in the adjusted dataset: 

```
stroke <- stroke %>%
  filter(gender != "Other") %>% # only one person in the entire dataset fits this criteria, I chose to drop them
  mutate(smoking_status = as.factor(smoking_status),
         bmi = as.numeric(bmi),
         work_type = as.factor(work_type),
         hypertension = as.factor(hypertension),
         Residence_type = as.factor(Residence_type),
         stroke = as.factor(stroke), 
         ever_married = as.factor(ever_married), 
         heart_disease = as.factor(heart_disease), 
         gender = as.factor(gender))

check_na <- function(x){
  aggr(x, col = c('navyblue','yellow'),
       numbers = TRUE, sortVars = TRUE, 
       labels = names(x), cex.axis = .7,
       gap = 2, ylab = c("Missing Data", "Pattern"))
}

check_na(stroke)
```

![image](https://user-images.githubusercontent.com/91495866/139962078-440d1be9-e683-4069-ac8b-b0f783b4faab.png)

Only one variable **bmi** had missing variables (approximately 3%), so I chose to deal with this using imputation.

```
imput <- mice(stroke, maxit=0)

meth = imput$method
predM = imput$predictorMatrix

predM[, c("bmi")] = 0 

set.seed(103)
imputed <- mice(stroke, method = meth, predictorMatrix = predM, m = 5)

imputed <- complete(imputed)


stroke <- imputed

remove(imputed)

check_na(stroke)
```

