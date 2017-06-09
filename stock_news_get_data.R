source('lib.R')

news <- read.csv('RedditNews.csv')
stocks <- read.csv('DJIA_table.csv')

news <- aggregate(News ~ Date, data = news, FUN = paste, collapse = '#####')
news <- separate(news, 'News', as.character(c(1:25)), sep = '#####')
news[is.na(news)] <- ''
news[2:26] <- as.data.frame(sapply(news[2:26], cleanString))
news[2:26] <- as.data.frame(lapply(news[2:26], tolower))

differences <- sign(-1 * diff(stocks$Open))
differences[differences == 0] <- 1
differences[differences == -1] <- 0

stocks <- data.frame(Date = stocks$Date[2:length(stocks$Date)], Change = differences)

stockNews <- merge(news, stocks, by = 'Date')

train <- stockNews[as.character(stockNews$Date) <= '2014-12-31', ]
test <- stockNews[as.character(stockNews$Date) > '2014-12-31', ]