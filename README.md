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

First I loaded relevant packages: 

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
```
