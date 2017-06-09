stopWords <- c("a","able","about","across","after","all","almost","also","am","among",
                "an","and","any","are","as","at","be","because","been","but","by","can",
                "cannot","could","dear","did","do","does","either","else","ever","every",
                "for","from","get","got","had","has","have","he","her","hers","him","his",
                "how","however","i","if","in","into","is","it","its","just","least","let",
                "like","likely","may","me","might","most","must","my","neither","no","nor",
                "not","of","off","often","on","only","or","other","our","own","rather","said",
                "say","says","she","should","since","so","some","than","that","the","their",
                "them","then","there","these","they","this","tis","to","too","twas","us",
                "wants","was","we","were","what","when","where","which","while","who",
                "whom","why","will","with","would","yet","you","your","aint","arent",
                "cant","couldve","couldnt","didnt","doesnt","dont","hasnt","hed",
                "hell","hes","howd","howll","hows","id","ill","im","ive","isnt"
                ,"its","mightve","mightnt","mustve","mustnt","shant","shed","shell",
                "shes","shouldve","shouldnt","thatll","thats","theres","theyd",
                "theyll","theyre","theyve","wasnt","wed","well","were","werent",
                "whatd","whats","whend","whenll","whens","whered","wherell",
                "wheres","whod","wholl","whos","whyd","whyll","whys","wont",
                "wouldve","wouldnt","youd","youll","youre","youve")

accuracy <- function(name, model, xtest, ytest) {
  pred = prediction(as.numeric(predict(model, xtest)), as.numeric(ytest))
  auc = performance(pred, measure = "auc")@y.values[[1]]
  print(paste('AUC for ', name, ' = ', auc))
  auc
}

cleanString <- function(string) {
  gsub(
    '\\d+',
    '',
    gsub(
      '[[:punct:]]',
      ' ',
      gsub(
        '[\'\"]',
        '',
        gsub(
         '(b\')|(b\")',
         '',
         string
        )
      )
    )
  )
}

getWordVectors <- function(text, window, minTerm, size, itNum) {
  tokens <- space_tokenizer(text)
  it <- itoken(tokens, progressbar = FALSE)
  vocab <- create_vocabulary(it, stopwords = stopWords)
  
  vocab <- prune_vocabulary(vocab, term_count_min = minTerm)
  vectorizer <- vocab_vectorizer(vocab, grow_dtm = FALSE, skip_grams_window = window)
  tcm <- create_tcm(it, vectorizer)
  
  glove <- GlobalVectors$new(word_vectors_size = size, vocabulary = vocab, x_max = 10)
  glove$fit(tcm, n_iter = itNum)
  
  glove$get_word_vectors()
}

wordsToVectors <- function(string, wordVectors) {
  words = lapply(
    Filter(
      function(word) word != "",
      strsplit(string, ' ', T)[[1]] 
    ),
    function(word) {
      if (word %in% dimnames(wordVectors)[[1]])
        wordVectors[word, , drop = F]
      else
        numeric(length(wordVectors[1,]))
    }
  )
  
  summary = Reduce(
    function(a,b) a + b,
    words
  )/length(words)
  
  if (length(summary) == 0) 0 else summary
}