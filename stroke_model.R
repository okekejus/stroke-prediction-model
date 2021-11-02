# Stroke prediction model 
# using the stroke/diabetes data, create a model that allows for prediciton of probability of stroke 
# based on individual lifestyle. 

# Stroke data: https://www.kaggle.com/thomaskonstantin/analyzing-and-modeling-stroke-data



#  Relevant Packages --------------------------------------------------------------------------------------------------------

library(pacman)

p_load(tidyverse, 
       ggthemes, 
       caret, 
       mice, 
       VIM, 
       rpart, 
       rpart.plot, 
       effects, 
       ggpubr, 
       ROSE)

# Load Data -----------------------------------------------------------------------------------------------------------------

stroke <-  read_csv("data/healthcare-dataset-stroke-data.csv") %>%
  select(-c(id))

str(stroke)

diabetes <- read_csv("data/diabetic_data.csv")

str(diabetes)



# Stroke Analysis -----------------------------------------------------------------------------------------------------------

stroke <- stroke %>%
  filter(gender != "Other") %>% # remove bc its just one person 
  mutate(smoking_status = as.factor(smoking_status),
         bmi = as.numeric(bmi),
         work_type = as.factor(work_type),
         hypertension = as.factor(hypertension),
         Residence_type = as.factor(Residence_type),
         stroke = as.factor(stroke), 
         ever_married = as.factor(ever_married), 
         heart_disease = as.factor(heart_disease), 
         gender = as.factor(gender))

  
# Missing Values 
check_na <- function(x){
  aggr(x, col = c('navyblue','yellow'),
       numbers = TRUE, sortVars = TRUE, 
       labels = names(x), cex.axis = .7,
       gap = 2, ylab = c("Missing Data", "Pattern"))
}

check_na(stroke)

# Imputation 
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




# Visualizations
# gender 

gender.dist <- tibble(gender = c("Female", "Male"), 
                      total = as.numeric(table(stroke$gender))) %>%
  mutate(gender.percent = round((total/sum(total) * 100))) %>%
  ggplot(aes(x = "", y = gender.percent, fill = gender)) + 
  geom_bar(stat = "identity", color = "black") + 
  coord_polar("y", start = 0) + 
  theme_void() + 
  scale_fill_manual(values = c("#E0BBE4", "#FFDFD3", "#957DAD")) + 
  geom_text(aes(label = paste0(gender.percent, "%")))

gender.dist


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
  geom_text(aes(label = paste0(stroke.percent, "%")))

outcome.dist # very imbalanced dataset, we will need to deal with this issue by upscaling

# age
age.dist <- stroke %>%
  ggplot(aes(x = age)) + 
  geom_histogram(binwidth = 10, fill = "#E0BBE4", color = "black") + 
  facet_wrap(~gender) +
  theme_hc()

age.dist


# Display all three things at once (outcomes, gender and age distributions)
ggarrange(outcome.dist, gender.dist, age.dist, 
          labels = c("1", "2", "3"), 
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
            geom_bar(stat = "identity", color = "black") + 
            coord_polar("y", 0) + 
            theme_void() + 
            scale_fill_manual(values = c("#B5D8D6", "#CAABD5")) + 
            geom_text(aes(label = paste0( gender.percent, "%")) ), 
          nrow = 2, ncol = 2, 
          labels = c("1", "2", "3"))





# Stroke Model --------------------------------------------------------------------------------------------------------------


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


# now make a binary tree model and visualize it 
stroke.tree <- rpart(stroke ~ ., data = stroke.trainTransformed, method = "class")
rpart.plot(stroke.tree)


# making a prediction 
stroke.tree.pred <- predict(stroke.tree, newdata = stroke.testTransformed)
roc.curve(stroke.testTransformed$stroke, stroke.tree.pred[, 2])


# testing prediction accuracy 
holdout <- ROSE.eval(stroke ~., data = stroke.trainTransformed, learner = rpart, 
                                     extr.pred = function(obj)obj[,2], seed = 1)

holdout$acc

ROSE.eval(stroke ~., data = stroke.testTransformed, learner = rpart, 
          extr.pred = function(obj)obj[,2], seed = 1)$acc


# Logistic regression (with caret)
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

densityplot(stroke.glm3)



# Logistic regression (with base)
mylogit <- glm(stroke ~ age + hypertension + avg_glucose_level + smoking_status,
               data = stroke.trainTransformed, family = "binomial")

summary(mylogit)

pred <- predict(mylogit, data = stroke.trainTransformed)
pred1 <- ifelse(pred > 0.5, 1, 0)
tab1 <- table(Predicted = pred1, Actual = stroke.trainTransformed$stroke)

1 - sum(diag(tab1))/sum(tab1)

with(mylogit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = F))

plot(allEffects(mylogit))


# Test Stroke Prediction  ---------------------------------------------------------------------------------------------------

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

# Testing with the tree
pred <- predict(stroke.tree, newdata = new_datTransformed)[, 2]


# Testing with the linear model 
pred <- predict(stroke.glm, newdata = new_datTransformed)
pred


pred <- predict(mylogit, newdata = new_datTransformed, type = "response")

pred
