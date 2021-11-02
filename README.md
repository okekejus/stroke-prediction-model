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
       ggpubr # to combine graphs 
       ROSE # synthetic data generation)

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

remove(imputed, meth, predM, imput)

check_na(stroke)
```
And we have a complete set now: 
![image](https://user-images.githubusercontent.com/91495866/139962391-8f442273-a73e-4832-84d2-e4a234198109.png)


## Step 2: Visualisations 
I used a few graphs to get some insight on the data. This was the code used for each one: 

```
# gender 

gender.dist <- tibble(gender = c("Female", "Male"), 
                      total = as.numeric(table(stroke$gender))) %>%
  mutate(gender.percent = round((total/sum(total) * 100), 3)) %>%
  ggplot(aes(x = "", y = gender.percent, fill = gender)) + 
  geom_bar(stat = "identity") + 
  coord_polar("y", start = 0) + 
  theme_void() + 
  scale_fill_manual(values = c("#E0BBE4", "#FFDFD3", "#957DAD")) + 
  geom_text(aes(label = gender.percent))

gender.dist

# Other, included in the non stroke pop. Will be removing this
stroke <- stroke %>%
  filter(gender != "Other") %>%
  tibble()

# How many people in the dataset had a stroke
outcomes <- tibble(outcome = c("No Stroke", "Stroke"), 
                   total = as.integer(table(stroke$stroke)))


outcome.dist <- outcomes %>%
  mutate(stroke.percent = round((total/sum(total)) * 100)) %>% 
  mutate(ypos = cumsum(stroke.percent) - 0.5*stroke.percent) %>%
  ggplot(aes(x = "", y = total, fill = outcome)) + 
  geom_bar(stat = "identity", color = "black") + 
  coord_polar("y", start = 0) + 
  scale_fill_manual(values = c("#E0BBE4", "#FFDFD3")) +
  theme_void() +  
  geom_text(aes(label = paste0(stroke.percent, "%"))) + 
  labs(title = "Proportion of stroke occurrences")

outcome.dist # very imbalanced dataset, we will need to deal with this issue by upscaling

# age
age.dist <- stroke %>%
  ggplot(aes(x = age)) + 
  geom_histogram(binwidth = 10, fill = "#E0BBE4", color = "black") + 
  facet_wrap(~gender) +
  theme_hc()

age.dist

# gender distribution 

# Display all three things at once (outcomes, gender and age distributions)
ggarrange(outcome.dist, gender.dist, age.dist, 
          labels = c("A", "B"), 
          ncol = 2, 
          nrow = 2)



ggarrange(stroke %>%
            ggplot() + 
            geom_jitter(aes(y = bmi, x = avg_glucose_level, color = stroke)) + 
            scale_color_manual(values = c("#93A7B7", "#FF6961")) + 
            labs(x = "Average Glucose Level"), 
          stroke %>%
            ggplot() + 
            geom_jitter(aes(y = bmi, x = age, color = stroke)) + 
            scale_color_manual(values = c("#93A7B7", "#FF6961")) + 
            labs(x = "Age"), 
          ncol = 1, 
          nrow = 2)


# High blood glucose and high ages tend to have more occurrences of strokes according to the graph.

stroke %>% 
  filter(stroke == 1) %>%
  ggplot(aes(x = age, y = avg_glucose_level)) +
  geom_jitter(color = "#FF6961")



ggarrange(stroke %>%
            filter(stroke == 1) %>%
            ggplot(aes(x = age)) + 
            geom_histogram(binwidth = 10, color = "black", fill = "#B5D8D6") + 
            facet_wrap(~gender) +
            theme_hc(),
          stroke %>%
            filter(stroke == 1) %>%
            ggplot(aes(x = bmi)) + 
            geom_histogram(binwidth = 10, color = "black", fill = "#B5D8D6") + 
            facet_wrap(~gender) + 
            theme_hc(), 
          tibble(gender = c("Female", "Male"), 
                 total = as.numeric(table(stroke$gender))) %>%
            mutate(gender.percent = round((total/sum(total) * 100), 1)) %>%
            ggplot(aes(x = "", y = gender.percent, fill = gender)) + 
            geom_bar(stat = "identity") + 
            coord_polar("y", 0) + 
            theme_void() + 
            scale_fill_manual(values = c("#B5D8D6", "#CAABD5")) + 
            geom_text(aes(label = gender.percent) ), 
          nrow = 2, ncol = 2)

```

Now here are the graphs generated: 

### Demographics (Overall)
![image](https://user-images.githubusercontent.com/91495866/139963048-d7a85bbd-8190-4186-b7ea-cc3d8e5e8977.png)

1. Only 5% of observations in the dataset went on to develop a stroke. This is a very imbalanced dataset. 
2. There are more obervations of females than makes in this dataset. 
3. Both age distributions for both genders are fairly normal, with wide tails as well. 

### Clinical Metrics 
![image](https://user-images.githubusercontent.com/91495866/139963185-c2a548c2-dc86-491c-8b25-47f4116f80b3.png)

According to these graphs, it seems the stroke population is largely within individuals age 40+, with high average glucose levels. For a better visualization of this population, I plotted age by glucose level for the red dots seen on the plot, to get this figure: 

![image](https://user-images.githubusercontent.com/91495866/139963327-631171b7-0dd5-4efe-89bd-3be08d6bdcc8.png)

And the associated code: 
```
stroke %>% 
  filter(stroke == 1) %>%
  ggplot(aes(x = age, y = avg_glucose_level)) +
  geom_jitter(color = "#FF6961")

```
### Demographics (Stroke Positive) 
![image](https://user-images.githubusercontent.com/91495866/139963619-9f540a4a-f427-4203-ae64-dffdccd248e2.png)

1. The stroke population is mostly older than age 50 for both genders. 
2. Bmi distribution is positively skewed, meaning the population doesn't consist of too many extreme bmis. 
3. There are more females than males in the stroke population as well. 


## Model Building 
I decided to try three models to see which one gave the most accurate results (and which method I preferred). The datasets were split using Caret and synthetic data generation was used to solve the imbalance in outcomes. Data was also scaled and centred to increase model accuracy (i.e mean of 0, standard deviation of 1). 

```
# Split Data 
set.seed(200)
ind <- createDataPartition(stroke$stroke, p = .8, times = 1)$Resample1

stroke.train <- stroke[ind, ]
stroke.test <- stroke[- ind, ]

preProcValues <- preProcess(stroke.train, method = c("center", "scale")) # modelling transformation using the training data
# doing this because refression methods work better if the input is standardized. 


stroke.trainTransformed <- predict(preProcValues, stroke.train)
stroke.testTransformed <- predict(preProcValues, stroke.test)

# very uneven distribution, need to check 
table(stroke.trainTransformed$stroke)
table(stroke.testTransformed$stroke)

# Using ROSE for synthetic data generation (due to imbalanced nature of outcomes)
set.seed(123)
stroke.trainTransformed <- ROSE(stroke ~ ., data = stroke.trainTransformed)$data

```

Prior to using ROSE, this was the distribution of the data: 
![image](https://user-images.githubusercontent.com/91495866/139964553-b65aa212-5c76-497f-b7e4-192dd8b6fc26.png)

After using it, this was the new distribution: 
![image](https://user-images.githubusercontent.com/91495866/139964612-cea683f0-4747-4f72-96d9-1f90beb7d883.png)



I used a binary regression tree model, as well as two logistic regression models (caret and base R). The results are below: 


## Binary Tree
This was pretty easy to do, with only a few line of code applied to the transformed data. 
```
stroke.tree <- rpart(stroke ~ ., data = stroke.trainTransformed, method = "class")
rpart.plot(stroke.tree) # to show the tree
```
![image](https://user-images.githubusercontent.com/91495866/139964727-381c0301-e474-444c-b6ac-4823b0fed447.png)

According to the tree, age and average glucose level seem to be the most important factors in determining an individual's stroke risk. The values present in the ages and glucose levels are the centred and standardized versions of the original dataset. 

To determine the model's accuracy, I made a prediction and plotted an ROC curve, as well as computing the accuracy using the holdout method. 

```
stroke.tree.pred <- predict(stroke.tree, newdata = stroke.testTransformed)
roc.curve(stroke.testTransformed$stroke, stroke.tree.pred[, 2])


# testing prediction accuracy 
holdout <- ROSE.eval(stroke ~., data = stroke.trainTransformed, learner = rpart, 
                                     extr.pred = function(obj)obj[,2], seed = 1)

holdout$acc


```

![image](https://user-images.githubusercontent.com/91495866/139965141-9a7c3390-754a-4da1-98db-de97e529d3c3.png)

The area under the curve is 0.804, and the accuracy of predictions with the training set was 80%. I decided to test the predicitons on the test data: 

```
ROSE.eval(stroke ~., data = stroke.testTransformed, learner = rpart, 
          extr.pred = function(obj)obj[,2], seed = 1)$acc
```

This returned an accuracy of 77%. 

## Logistic Regression (Caret)
Using caret, I ran a Bayesian generalized linear model on the training data used for the binary tree as well. It took a couple tries to get to an optimum accuracy, leading to the creation of a few models (for comparison). The final model is outlined in the code: 

```
fitControl <- trainControl(method = "boot", 
                           number = 30)


stroke.glm <- train(stroke ~., data = stroke.trainTransformed, 
                    method = "bayesglm", 
                    trControl = fitControl)

stroke.glm.summary <- summary(stroke.glm)

stroke.glm.summary

pred <- predict(stroke.glm, newdata = stroke.testTransformed)


postResample(pred = pred, obs = stroke.testTransformed$stroke) # accuracy test 

stroke.glm2 <- train(stroke ~ age + hypertension + smoking_status + avg_glucose_level + 
                       bmi, data = stroke.trainTransformed, 
                    method = "bayesglm", 
                    trControl = fitControl)

pred <- predict(stroke.glm2, newdata = stroke.testTransformed)


postResample(pred = pred, obs = stroke.testTransformed$stroke) # very slightly more accurate 



stroke.glm3 <- train(stroke ~ age + hypertension + smoking_status + avg_glucose_level, 
                     data = stroke.trainTransformed, 
                     method = "bayesglm", 
                     trControl = fitControl)


summary(stroke.glm3) # final model 

pred <- predict(stroke.glm3, newdata = stroke.testTransformed)


postResample(pred = pred, obs = stroke.testTransformed$stroke)


stroke.glm4 <- train(stroke ~ age + hypertension + avg_glucose_level, 
                     data = stroke.trainTransformed, 
                     method = "bayesglm", 
                     trControl = fitControl)

summary(stroke.glm4)

pred <- predict(stroke.glm4, newdata = stroke.testTransformed)


postResample(pred = pred, obs = stroke.testTransformed$stroke) # less accurate than when hypertension is included
```

Accuracy went from 73.2% (model 1) to 73.6% (model 2) to 74% (model 3) and then back down to 73.5% (model 4). I decided to go with the third model, which included age, hypertension, smoking status and average glucose level as coefficients. 


# Logistic Regression (base R) 
I decided to use base R, as I prefer to visualize effects with this due to its ease.

I recreated the model determined from above in base R: 

```
mylogit <- glm(stroke ~ age + hypertension + avg_glucose_level + smoking_status,
               data = stroke.trainTransformed, family = "binomial")
               
 # testing predictions 
 pred <- predict(mylogit, data = stroke.trainTransformed)
pred1 <- ifelse(pred > 0.5, 1, 0)
tab1 <- table(Predicted = pred1, Actual = stroke.trainTransformed$stroke) # creating confusion matrix 

1 - sum(diag(tab1))/sum(tab1) # confusion matrix evaluation 

with(mylogit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = F)) # calculating goodness of fit 
```

This model proved to be slightly more accurate than the caret model by 1%, with an accuracy of 75%. 

I also created a plot of the effects to see how they interact with the outcome: 

```
plot(allEffects(mylogit))
```

![image](https://user-images.githubusercontent.com/91495866/139966434-50dca7da-8c18-4d61-b842-1864e8f340d1.png)

The graphs are pretty clear: 
- As one ages, risk of stroke increases 
- As glucose level increases, so does risk of stroke. 
- Non smokers have a lower risk of developing a stroke vs current and former smokers. 
- Previous incidence of hypertension increases stroke risk. 

## Predicting Stroke Risk 
Ok now the fun part! I decided to make predictions for my risk of a stroke with all three models, to see if their responses would differ (and by how much). 

### Binary Tree
```
names(stroke.testTransformed)

new_dat <- tibble(gender = "Male", 
                  age = 40,
                  hypertension = "0",
                  heart_disease = "0", 
                  ever_married = "No",
                  work_type = "Private", 
                  Residence_type =  "Urban", 
                  avg_glucose_level = 178.2, 
                  bmi = 24.9,
                  smoking_status = "never smoked")
new_datTransformed <- predict(preProcValues, new_dat)

# Testing with the tree
pred <- predict(stroke.tree, newdata = new_datTransformed)[, 2]

```
The tree concluded I currently have an 11% chance of developing a stroke (This method provides a probability for developing or not developing a stroke, of which I scored 0.11 in the "developing" category). 


# With Logistic Regression
```
names(stroke.testTransformed)

new_dat <- tibble(gender = "Male", 
                  age = 23,
                  hypertension = "0",
                  heart_disease = "0", 
                  ever_married = "No",
                  work_type = "Private", 
                  Residence_type =  "Urban", 
                  avg_glucose_level = 178.2, 
                  bmi = 24.9,
                  smoking_status = "never smoked")
new_datTransformed <- predict(preProcValues, new_dat)

# Testing with caret

pred <- predict(stroke.glm, newdata = new_datTransformed)

# Testing with base R
pred <- predict(mylogit, newdata = new_datTransformed, type = "response")
```
The caret model gives  a value of 0 or 1, corresponding to stroke/no stroke. I recieved a 0 as my output. 

The base R model gives a probability of being categorized as a stroke patient. Mine was 0.22 - slightly higher than the probability for the tree model, but this is expected as it is a less accurate model. The conclusion remains the same however - I am not at risk of developing a stroke yet. 
