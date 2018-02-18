#last.updated 11/17 23:37 by Ruyyi

#retrieve data from the original file
data = read.table('/Users/ruyyi/Documents/Course/DS502/Project/ratings.csv',sep = ',',header = TRUE)
userID = data$userId
movieID  = data$movieId
userID.list = unique(sort(data$userId))
movieID.list = unique(sort(data$movieId))
size.user = length(unique(userID))
size.movie = length(unique(movieID))
size.data = nrow(data)
userID.index = match(data$userId,userID.list)
movieID.index = match(data$movieId,movieID.list)

#setting up coefficients
k = 15
fold = 10

#building k-fold validation matrix
kfold.rmse.matrix = matrix(0,fold,2)

#Randomly shuffle the data
data<-data[sample(1:size.data,size.data),]

#Create 10 equally size folds
folds <- cut(seq(1,size.data),breaks=10,labels=FALSE)

fold.flag = 1

kfold.indexes <- which(folds==fold.flag,arr.ind=TRUE)
test.data <- data[kfold.indexes, ]
train.data <- data[-kfold.indexes, ]

#get statistical information
size.train.data = nrow(train.data)
size.test.data = nrow(test.data)

train.userID.index = match(train.data$userId,userID.list)
train.movieID.index = match(train.data$movieId,movieID.list)
test.userID.index = match(test.data$userId,userID.list)
test.movieID.index = match(test.data$movieId,movieID.list)

#building the rating matrix
train.rating.matrix = matrix(0,size.user,size.movie)
test.rating.matrix = train.rating.matrix
#building predicition rating comparison matrix
pred.rating.log.matrix = matrix(0,size.test.data,5)

#building rating matrix
for (rate in 1:size.train.data)
{
  train.rating.matrix[train.userID.index[rate],train.movieID.index[rate]] = train.data$rating[rate]
}

for (rate in 1:size.test.data)
{
  test.rating.matrix[test.userID.index[rate],test.movieID.index[rate]] = test.data$rating[rate]
}

#cosine similarity computation
Mod.movie = colSums(train.rating.matrix^2)^0.5
Mod.user = colSums(t(train.rating.matrix)^2)^0.5

temp.movie.matrix.Numer = crossprod(train.rating.matrix)
temp.user.matrix.Numer = crossprod(t(train.rating.matrix))

temp.movie.matrix.Denor = Mod.movie %*% t(Mod.movie)
temp.user.matrix.Denor = Mod.user %*% t(Mod.user)

temp.movie.matrix.Denor[temp.movie.matrix.Denor == 0] = Inf
temp.user.matrix.Denor[temp.user.matrix.Denor == 0] = Inf

Correlation.movie = temp.movie.matrix.Numer / temp.movie.matrix.Denor
Correlation.user = temp.user.matrix.Numer / temp.user.matrix.Denor
diag(Correlation.movie) = 1
diag(Correlation.user) = 1

#compute the average similarity of user and movie
test.averagesim.matrix = matrix(0,size.test.data,12)
colnames(test.averagesim.matrix) = c('user.index','user.ratenum','user.mean','user.variance','user.avgsim','user.prederr','movie.index','movie.ratednum','movie.mean','movie.variance','movie.avgsim','movie.prederr')

#Prediction part start
for (rate in 1:size.test.data)
{
  #Set up a matrix to store the k neighbors information and temporary neighbors information
  temp.neighbor.movie = matrix(0,size.movie-1,3)
  temp.neighbor.user = matrix(0,size.user-1,3)
  
  #fetch test userID index and movieID index
  pred.user.index = test.userID.index[rate]
  pred.movie.index = test.movieID.index[rate]
  #fetch user neighbors
  #first column implies the index of neighbor userID
  #second column implies the correspond similarity
  #third column implies the rating of correspond user to the unpredicted movie
  temp.neighbor.user[,2] = sort(Correlation.user[,pred.user.index],decreasing = TRUE)[-1]
  temp.neighbor.user[,1] = order(Correlation.user[,pred.user.index],decreasing = TRUE)[-1]
  for (neighbor in 1:dim(temp.neighbor.user)[1])
  {
    temp.neighbor.user[neighbor,3] = train.rating.matrix[temp.neighbor.user[neighbor,1],pred.movie.index]
  }
  #fetch user neighbors
  #first column implies the index of neighbor userID
  #second column implies the correspond similarity
  #third column implies the rating of correspond user to the unpredicted movie
  temp.neighbor.movie[,2] = sort(Correlation.movie[,pred.movie.index],decreasing = TRUE)[-1]
  temp.neighbor.movie[,1] = order(Correlation.movie[,pred.movie.index],decreasing = TRUE)[-1]
  for (neighbor in 1:dim(temp.neighbor.user)[1])
  {
    temp.neighbor.movie[neighbor,3] = train.rating.matrix[pred.user.index,temp.neighbor.movie[neighbor,1]]
  }
  #eliminate the 0-rating and 0-similarity neighbors from the neighbor list
  temp.neighbor.movie = temp.neighbor.movie[which(temp.neighbor.movie[,3] != 0 & temp.neighbor.movie[,2] != 0),]
  temp.neighbor.user = temp.neighbor.user[which(temp.neighbor.user[,3] != 0 & temp.neighbor.user[,2] != 0),]
  temp.neighbor.movie = matrix(temp.neighbor.movie,ncol = 3)
  temp.neighbor.user = matrix(temp.neighbor.user,ncol = 3)
  
  #if there exist neighbors, make the predictions, else, set the prediction score as the average rate.
  if (dim(temp.neighbor.movie)[1] == 0)
  {
    temp.rating.vector = (train.rating.matrix[pred.user.index,])
    if (sum(temp.rating.vector) == 0)
    {
      pred.rating.itembased = 3
    } else {
      pred.rating.itembased = mean(temp.rating.vector[which(temp.rating.vector != 0)])
    }
  } else {
    #pick the top-k neighbor if they have
    #neighbor.movie = matrix(0,min(k,dim(temp.neighbor.movie)[1]),3)
    neighbor.movie = matrix(temp.neighbor.movie[1:min(k,dim(temp.neighbor.movie)[1]),], ncol = 3)
    #similarity fused part (working on)
    #neighbor.simfuse.sim = Correlation.movie[neighbor.user[,1],neighbor.movie[,1]]
    #neighbor.simfuse.rate = train.rating.matrix[neighbor.user[,1],neighbor.movie[,1]]
    pred.rating.itembased = as.numeric((t(neighbor.movie[,2]) %*% neighbor.movie[,3])) / sum(neighbor.movie[,2])
  }
  
  if (dim(temp.neighbor.user)[1] == 0)
  {
    temp.rating.vector = (train.rating.matrix[,pred.user.index])
    if (sum(temp.rating.vector) == 0)
    {
      pred.rating.userbased = 3
    } else {
      pred.rating.userbased = mean(temp.rating.vector[which(temp.rating.vector != 0)])
    }
  } else {
    #neighbor.user = matrix(0,min(k,dim(temp.neighbor.user)[1]),3)
    neighbor.user = matrix(temp.neighbor.user[1:min(k,dim(temp.neighbor.user)[1]),], ncol = 3)
    #similarity fused part (working on)
    #neighbor.simfuse.sim = Correlation.movie[neighbor.user[,1],neighbor.movie[,1]]
    #neighbor.simfuse.rate = train.rating.matrix[neighbor.user[,1],neighbor.movie[,1]]
    pred.rating.userbased = as.numeric((t(neighbor.user[,2]) %*% neighbor.user[,3])) / sum(neighbor.user[,2])
  }
  #print(c(pred.rating.itembased, pred.rating.userbased, test.rating.matrix[pred.user.index,pred.movie.index]))
  pred.rating.log.matrix[rate,1] = pred.user.index
  pred.rating.log.matrix[rate,2] = pred.movie.index
  pred.rating.log.matrix[rate,3] = pred.rating.itembased
  pred.rating.log.matrix[rate,4] = pred.rating.userbased
  pred.rating.log.matrix[rate,5] = test.rating.matrix[pred.user.index,pred.movie.index]
  
  #compute the average similarity of user and movie, compare the statistical informations
  test.averagesim.matrix[rate,1] = pred.user.index
  test.averagesim.matrix[rate,2] = length(train.rating.matrix[pred.user.index,(which(train.rating.matrix[pred.user.index,] != 0))])
  test.averagesim.matrix[rate,3] = mean(train.rating.matrix[pred.user.index,(which(train.rating.matrix[pred.user.index,] != 0))])
  test.averagesim.matrix[rate,4] = var(train.rating.matrix[pred.user.index,(which(train.rating.matrix[pred.user.index,] != 0))])
  test.averagesim.matrix[rate,5] = mean(neighbor.user[,2])
  test.averagesim.matrix[rate,6] = pred.rating.userbased - test.rating.matrix[pred.user.index,pred.movie.index]
  test.averagesim.matrix[rate,7] = pred.movie.index
  test.averagesim.matrix[rate,8] = length(train.rating.matrix[(which(train.rating.matrix[,pred.movie.index] != 0)),pred.movie.index])
  test.averagesim.matrix[rate,9] = mean(train.rating.matrix[(which(train.rating.matrix[,pred.movie.index] != 0)),pred.movie.index])
  test.averagesim.matrix[rate,10] = var(train.rating.matrix[(which(train.rating.matrix[,pred.movie.index] != 0)),pred.movie.index])
  test.averagesim.matrix[rate,11] = mean(neighbor.movie[,2])
  test.averagesim.matrix[rate,12] = pred.rating.itembased - test.rating.matrix[pred.user.index,pred.movie.index]
}
pairs((test.averagesim.matrix[,-1])[,-6])
hist(test.averagesim.matrix[,5],test.averagesim.matrix[,7])
histogram = hist(test.averagesim.matrix[,5],ylab = 'frequency',xlab='average similarity',main = 'testset user average similarity histogram')
#RMSE calculation
RMSE.userbased = sqrt(sum((pred.rating.log.matrix[,4]-pred.rating.log.matrix[,5])^2) / nrow(pred.rating.log.matrix))
RMSE.itembased = sqrt(sum((pred.rating.log.matrix[,3]-pred.rating.log.matrix[,5])^2) / nrow(pred.rating.log.matrix))