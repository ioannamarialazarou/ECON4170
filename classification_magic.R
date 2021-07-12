classification_magic <- function(data, normalization = 1, split = 0.80, fs = 1, balance = 1, classifier = 1, tune = 0) {
        
        # Min-max normalization 
        if (normalization == 1) {
                data_normalized <- as.data.frame(lapply(data[,c(1:(dim(data)[2]-1))], normalized))
                data_normalized = cbind(data_normalized,data[,dim(data)[2]])
                colnames(data_normalized)[dim(data)[2]] = 'class'
                data = data_normalized
        }
        
        # Set the split threshold
        sep <- floor(split * nrow(data))
        
        # Perform data sampling to improve generalization
        # https://www.quora.com/What-is-generalization-in-machine-learning
        train_ind <- sample(seq_len(nrow(data)), size = sep)
        
        train <- data[train_ind, ] 
        test <- data[-train_ind, ] 
        # dim(train)
        # dim(test)
        
        # Feature Selection
        # https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
        
        if (fs == 1) { # Correlation
                # Calculate correlation matrix (except the target/categorical value)
                correlationMatrix <- cor(train[,1:(dim(train)[2]-1)])
                # Summarize the correlation matrix
                print(correlationMatrix)
                # Find attributes that are highly corrected - set threshold to 0.75 - modify on demand
                highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
                # Remove highly correlated attributes based on the index
                train = train[,-highlyCorrelated]
                test = test[,-highlyCorrelated]
        } else if (fs == 2) { # Recursive Feature Elimination (RFE)
                # Define the control using a random forest selection function
                control <- rfeControl(functions=rfFuncs, method="cv", number=3)
                # Run the RFE algorithm
                results <- rfe(train[,1:(dim(train)[2]-1)], train[,dim(train)[2]], sizes=c(1:(dim(train)[2]-1)), rfeControl=control)
                # Summarize the results
                print(results)
                # List the chosen features
                predictors(results)
                # plot the results
                ggplot(results, type=c("g", "o")) + ggsave('rfe.pdf')
                
                train_t = train %>% select(predictors(results))
                test_t = test %>% select(predictors(results))
                train = cbind(train_t,train[,dim(train)[2]])  
                colnames(train)[dim(train)[2]] = 'class'
                test = cbind(test_t,test[,dim(test)[2]])  
                colnames(test)[dim(test)[2]] = 'class'
        } else if (rf == 3) { # Importance based on Learning Vector Quantization (LVQ)
                # Prepare training scheme
                control <- trainControl(method="repeatedcv", number=3, repeats=3)
                # train the model
                model <- train(class~., data=train, method="lvq", preProcess="scale", trControl=control)
                # estimate variable importance
                importance <- varImp(model, scale=FALSE)
                # summarize importance
                print(importance)
                # plot importance
                png('importance.png')
                plot(importance)
                dev.off()
                
                features = order(importance[[1]][1]$g,decreasing = TRUE)
                
                train_t = train[,features[1:3]]
                test_t = test[,features[1:3]]
                train = cbind(train_t,train[,dim(train)[2]])  
                colnames(train)[dim(train)[2]] = 'class'
                test = cbind(test_t,test[,dim(test)[2]]) 
                colnames(test)[dim(test)[2]] = 'class'
        } else { # Boruta # https://www.machinelearningplus.com/machine-learning/feature-selection/
                boruta_output <- Boruta(class ~ ., data=train, doTrace=0)  
                names(boruta_output)
                
                # Get significant variables including tentatives
                boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
                print(boruta_signif)  
                
                # Do a tentative rough fix
                roughFixMod <- TentativeRoughFix(boruta_output)
                boruta_signif <- getSelectedAttributes(roughFixMod)
                print(boruta_signif)
                
                # Variable Importance Scores
                imps <- attStats(roughFixMod)
                imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
                head(imps2[order(-imps2$meanImp), ])  # descending sort
                
                # Plot variable importance
                png('boruta.png')
                plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance") 
                dev.off()
                
                train_t = train[,rownames(head(imps2[order(-imps2$meanImp), ]))]
                test_t = test[,rownames(head(imps2[order(-imps2$meanImp), ]))]
                train = cbind(train_t,train[,dim(train)[2]])    
                colnames(train)[dim(train)[2]] = 'class'
                test = cbind(test_t,test[,dim(test)[2]])   
                colnames(test)[dim(test)[2]] = 'class'
        }
        
        # Balancing classes 
        
        if (balance == 1) { # undersampling the majority class
                class1 = train[train$class=='g',]
                class2 = train[train$class=='h',]
                
                class1_under = class1[1:dim(class2)[1],]
                train_balanced = rbind(class1_under,class2)
                
                class1 = test[test$class=='g',]
                class2 = test[test$class=='h',]
                
                class1_under = class1[1:dim(class2)[1],]
                test_balanced = rbind(class1_under,class2)
        } else if (balance == 2) { # oversampling the minority class by replicas
                class1 = train[train$class=='g',]
                class2 = train[train$class=='h',]
                
                # Find how many samples I need to match the majority class
                samples_to_generate = dim(class1)[1] - dim(class2)[1]
                # Select samples_to_generate random samples from the minority class
                random_samples = sample_n(class2,size = samples_to_generate)
                class2_over = rbind(class2,random_samples)

                train_balanced = rbind(class1,class2_over)
                
                class1 = test[test$class=='g',]
                class2 = test[test$class=='h',]
                
                # Find how many samples I need to match the majority class
                samples_to_generate = dim(class1)[1] - dim(class2)[1]
                # Select samples_to_generate random samples from the minority class
                random_samples = sample_n(class2,size = samples_to_generate)
                class2_over = rbind(class2,random_samples)
                
                test_balanced = rbind(class1,class2_over)
        } else { # SMOTE
                train_balanced <- SMOTE(class ~ ., train, perc.over = 100)
                test_balanced <- SMOTE(class ~ ., test, perc.over = 100)
        }
        train = train_balanced
        test = test_balanced
        
        # Machine Learning (Classification)
        # https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/
        
        # Define control (with tuning or not)
        if (tune == 0) {
                control <- trainControl(method="repeatedcv", number=5, repeats=3, classProbs = TRUE)
        } else {
                control <- trainControl(method="repeatedcv", number=5, repeats=3, classProbs = TRUE, search = 'random')
        }
                
        if (classifier == 1) { # Random Forests  https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
                metric <- "Accuracy"
                mtry <- sqrt(ncol(train))
                tunegrid <- expand.grid(.mtry=mtry)
                if (tune == 0) {
                        model <- train(class~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
                } else {
                        model <- train(class~., data=train, method="rf", metric=metric, trControl=control, tuneLength=5)
                }
        } else { # Support Vector Machines
                model <- train(class ~., data = train, method = "svmRadial", trControl=control, preProcess = c("center", "scale"))
        }
        print(model)
        
        predictions = predict(model, newdata = test)
        predictions_p <- predict(model, newdata = test, type = "prob")

        cm = confusionMatrix(predictions, test$class)
        
        #https://www.displayr.com/what-is-a-roc-curve-how-to-interpret-it/?utm_referrer=https%3A%2F%2Fwww.google.com%2F
        ROC <- roc(test$class, predictions_p[,2])
        ROC_auc <- auc(ROC)
        
        plot(ROC, col = "black", main = "ROC")
        
        # print the performance of each model
        message("Accuracy model %: ", format(model$results$Accuracy,digits = 4))
        message("Kappa model %: ", format(model$results$Kappa,digits = 4))
        
        message("Accuracy %: ", format(cm$overall[1],digits = 4))
        message("Sensitivity %: ", format(cm$byClass[1],digits = 4))
        message("Specificity %: ", format(cm$byClass[2],digits = 4))
        message("Precision %: ", format(cm$byClass[5],digits = 4))
        message("Recall %: ", format(cm$byClass[6],digits = 4))
        message("F1 %: ", format(cm$byClass[7],digits = 4))
        message("Area under curve: ", format(ROC_auc,digits = 4))
        
        message('Confusion matrix: ')
        print(cm$table)
        
        # heatmap(cm$table, col = rainbow(3),xlab = 'Reference', ylab = 'Predicted')
}

normalized <- function(x) { (x- min(x))/(max(x) - min(x)) }
