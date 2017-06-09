source('stock_news_get_data.R')

text <- Reduce(function(x,y) paste0(x,y), train[2:26])

result <- lapply(
  list(
    c(5,50),c(5,100),
    c(7,50),c(7,100),
    c(10,100),c(10,150),c(10,200),
    c(12,150),c(12,200),
    c(15,150),c(15,200)
  ),
  function(params) {
    print(paste('Skip grams window = ', params[1], ', ', 'Word vector size', params[2]))
    wordVectors <- getWordVectors(text, params[1], 10L, params[2], 20)
    wvLen <- length(wordVectors[1,])
    #wordVectors <- prcomp(wordVectors, scale. = T)$x
    
    preprocess <- function(data) {
      data <- data[c(1, 27, 2:26)]
      
      data[3:27] <- t(apply(
        data[3:27],
        2,
        function(column) lapply(
          column,
          function(string) wordsToVectors(string, wordVectors)
        )
      ))
      
      data[3:(wvLen+2)] <- t(apply(
        data[3:27], 
        1, 
        function(row) Reduce(
          function(a,b) a+b,
          row
        )/25
      ))
      
      data <- data[2:(wvLen+2)]
    }
    
    train <- preprocess(train)
    names(train) <- make.names(names(train))
    
    test <- preprocess(test)
    names(test) <- make.names(names(test))
    
    lR <- cv.glmnet(x = as.matrix(train[2:(wvLen+1)]), y = train$Change, family = 'binomial', alpha = 0,type.measure = "auc",nfolds = 4,thresh = 1e-3,maxit = 1e3)
    ac1 <- accuracy("LR", lR, as.matrix(test[2:(wvLen+1)]), test$Change)
    
    linSVM <- svm(as.matrix(train[2:(wvLen+1)]), y=train$Change, type = 'C', kernel = 'linear')
    ac2 <- accuracy("SVM", linSVM, test[2:(wvLen+1)], test$Change)
    
    rF <- randomForest(as.factor(Change) ~ ., data = train)
    ac3 <- accuracy("RF", rF, test[2:(wvLen+1)], test$Change)
    
    nB <- naiveBayes(as.factor(Change) ~ ., data = train)
    ac4 <- accuracy("NB", nB, test[2:(wvLen+1)], test$Change)
    
    xgb <- xgboost(data = as.matrix(train[2:(wvLen+1)]), label = train$Change, nthread = 2, max_depth = 2, nrounds = 200, objective = "binary:logistic", verbose = 0)
    ac5 <- accuracy("XGB", xgb, as.matrix(test[2:(wvLen+1)]), test$Change)
    
    list(ac1,ac2,ac3,ac4,ac5)
})

result <- as.data.frame(matrix(unlist(result), nrow=length(unlist(result[1]))))

colnames(result) <- c(
  '(5,50)','(5,100)',
  '(7,50)','(7,100)',
  '(10,100)','(10,150)','(10,200)',
  '(12,150)','(12,200)',
  '(15,150)','(15,200)'
)

rownames(result) <- c(
  'LR', 'SVM', 'RF', 'NB', 'XGB'
)

ngrams <- factor(colnames(result), levels = colnames(result))

par(mar = c(6.5, 6.5, 0.5, 0.5), mgp = c(4, 1, 0))
plot(ngrams, result[1,], axes=F, col="blue", 'l', xlab = 'Skip gram window, vector size', ylab='AUC',ylim=c(0.4,0.6))
axis(2)
axis(1, at=seq_along(result[1,]),labels=as.character(ngrams), las=2)
lines(ngrams, result[2,], col="red")
lines(ngrams, result[3,], col="green")
lines(ngrams, result[4,], col="black")
lines(ngrams, result[5,], col="purple")
legend('topright', rownames(result), lty=c(1,1), col=c('blue','red','green','black','purple'), ncol=2)

