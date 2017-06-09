source('stock_news_get_data.R')

train <- data.frame(train$Date, train$Change, Reduce(function(x,y) paste0(x,y), train[2:26]))
colnames(train) <- c('Date', 'Change', 'News')

test <- data.frame(test$Date, test$Change, Reduce(function(x,y) paste0(x,y), test[2:26]))
colnames(test) <- c('Date', 'Change', 'News')

result <- lapply(
  list(
    c(1L,1L),c(1L,2L),c(1L,3L),c(1L,4L),
    c(2L,2L),c(2L,3L),c(2L,4L),
    c(3L,3L),c(3L,4L),
    c(4L,4L)
  ),
  function(params) {
    print(paste('N-grams = ', params))
  
    trainIt <- itoken(as.character(train$News), preprocessor = tolower, tokenizer = word_tokenizer, ids = train$Date, progressbar = F)
    vocab <- create_vocabulary(trainIt, stopwords = stopWords, ngram=params)
    vocab <- prune_vocabulary(vocab, term_count_min = 10L, doc_proportion_max = 0.5)
    vectorizer <- vocab_vectorizer(vocab)
    trainDtm <- create_dtm(trainIt, vectorizer)
    
    testIt <- itoken(as.character(test$News), preprocessor = tolower, tokenizer = word_tokenizer, ids = test$Date, progressbar = F)
    testDtm <- create_dtm(testIt, vectorizer)
    
    tfidf <- TfIdf$new()
    trainDtmTfidf <- fit_transform(trainDtm, tfidf)
    testDtmTfidf <- transform(testDtm, tfidf)
    
    lR <- cv.glmnet(x = trainDtm, y = train$Change, 
                    family = 'binomial', 
                    alpha = 0,
                    type.measure = "auc",
                    nfolds = 4,
                    thresh = 1e-3,
                    maxit = 1e3)
    ac1 <- accuracy("LR", lR, testDtm, test$Change)
    
    lR2 <- cv.glmnet(x = trainDtmTfidf, y = train$Change, 
                     family = 'binomial', 
                     alpha = 0,
                     type.measure = "auc",
                     nfolds = 4,
                     thresh = 1e-3,
                     maxit = 1e3)
    ac2 <- accuracy("LR TFIDF", lR2, testDtmTfidf, test$Change)
    
    linSVM <- svm(trainDtm, y=train$Change, type = 'C', kernel = 'linear')
    ac3 <- accuracy("SVM", linSVM, testDtm, test$Change)
    
    linSVM <- svm(trainDtmTfidf, y=train$Change, type = 'C', kernel = 'linear')
    ac4 <- accuracy("SVM TFIDF", linSVM, testDtmTfidf, test$Change)
    
    xgb <- xgboost(data = trainDtm, label = train$Change, nthread = 2, max_depth = 2, nrounds = 120, objective = "binary:logistic", verbose = 0)
    ac5 <- accuracy("XGB", xgb, testDtm, test$Change)
    
    xgb <- xgboost(data = trainDtmTfidf, label = train$Change, max_depth = 2, nthread = 2, nrounds = 150, objective = "binary:logistic", verbose = 0)
    ac6 <- accuracy("XGB TFIDF", xgb, testDtmTfidf, test$Change)
    
    list(ac1,ac2,ac3,ac4,ac5,ac6)
})

result <- as.data.frame(matrix(unlist(result), nrow=length(unlist(result[1]))))

colnames(result) <- c(
  '1,1','1,2','1,3','1,4',
  '2,2','2,3','2,4',
  '3,3','3,4',
  '4,4'
)

rownames(result) <- c(
  'LR', 'LR Tfidf', 'SVM', 'SVM Tfidf', 'XGB', 'XGB Tfidf'
)

ngrams <- factor(colnames(result))

plot(ngrams, result[1,], axes=F, col="blue", 'l', xlab = 'ngrams', ylab='AUC',ylim=c(0.4,0.6))
axis(2)
axis(1, at=seq_along(result[1,]),labels=as.character(ngrams), las=2)
lines(ngrams, result[2,], col="red")
lines(ngrams, result[3,], col="green")
lines(ngrams, result[4,], col="black")
lines(ngrams, result[5,], col="purple")
lines(ngrams, result[6,], col="orange")
legend('topright', rownames(result), lty=c(1,1), col=c('blue','red','green','black','purple','orange'), ncol=2)
