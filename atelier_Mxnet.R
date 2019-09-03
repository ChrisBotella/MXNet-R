#cran <- getOption("repos")
#cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu80"
#options(repos = cran)
#install.packages("mxnet")
require(mxnet)
require(imager)
require(ggplot2)

#####
# Functions
#####


# Returns the list of derivative of the loss with respect to each parameter
# grouped and named by model parts. ex : "conv1_weight"
get.all.gradients = function(model,X,y,executor,X_name="data",y_name="label"){
  # on lui donne les param�tres du mod�le 
  mx.exec.update.arg.arrays(executor , model$arg.params,match.name = T )
  mx.exec.update.aux.arrays(executor, model$aux.params, match.name=T)
  # on lui donne les donn�es d'entr�e (et de sortie pour la loss)
  command = paste('mx.exec.update.arg.arrays(executor,list(',
                  X_name,
                  '=mx.nd.array(X),',
                  y_name,
                  '=mx.nd.array(y)),match.name=T)',
                  sep="")
  eval(parse(text= command))
  mx.exec.forward(executor)
  mx.exec.backward(executor)
  return(list(executor$grad.arrays[!names(executor$grad.arrays)%in%c(X_name,y_name)],executor$ref.aux.arrays))
}


# personal prediction service (compatible with 2D (vector) and 4D (3-dim arrays) input data, as well as uni and multilabel output)
predict.executor = function(predictor.symbol,params,Array,devices,max.compute.size=100,aux.params=NULL,verbose=F){
  if(length(dim(Array))==4){
    if(verbose){print('Assume colmajor, detect 4 dimensional input')}
    N = dim(Array)[4]
    first_dim = dim(Array)[1:3]
  }else{
    if(verbose){print('Assume colmajor, detect 2 dimensional input')}
    N = dim(Array)[2]
    first_dim = dim(Array)[1]
  }
  n.pass = - (-N %/% max.compute.size)
  rest = N %% max.compute.size
  
  # We compute predictions per data slice
  for(i in 1:n.pass){
    not.complete = rest>0 & i==n.pass
    # We create an executor for computing predictions on the data slice
    prediction = mx.simple.bind(predictor.symbol,
                                ctx=devices,
                                data= c(first_dim,not.complete * rest + (!not.complete) * max.compute.size))
    # We provide the input data to the executor
    if(not.complete){band=(N-rest+1):N}else{band = (i-1)*max.compute.size + 1:max.compute.size}
    # arguments parameters are the model weights
    if(length(dim(Array))==4){
      mx.exec.update.arg.arrays(prediction,c(list(data=mx.nd.array(Array[,,,band,drop=F])), params),match.name=T)
    }else{
      mx.exec.update.arg.arrays(prediction,c(list(data=mx.nd.array(Array[,band,drop=F])),params),match.name=T)
    }
    # auxiliary parameters are hyper-parameters for which gradient is not computed 
    # They may be used as global statistics, especially for adaptative SGD algorithms 
    if(!is.null(aux.params)){mx.exec.update.aux.arrays(prediction,aux.params,match.name=T)}
    
    
    # is.train=F must be specified for prediction
    # Batchnorm is then applied with the current global statistics of aux.params
    # and those aux.params are unaffected by the current mini-batch
    mx.exec.forward(prediction,is.train=F)
    if(i==1){
      # first prediction slice
      p = as.array(prediction$outputs[[1]])
    }else{
      # concatenate to previous slices
      if(is.null(dim(p))){
        p = c(p, as.array(prediction$outputs[[1]]) )
      }else{
        if(i==2 & verbose){print('detect multilabel, returns prediction matrix')}
        p = cbind( p , as.array(prediction$outputs[[1]]) )
      }
    }
    gc(reset=T)
  }
  return(p)
}

# Loss executor
loss.executor = function(predictor.symbol,params,Array,y,devices,max.compute.size=100,aux.params=NULL,verbose=F){
  if(length(dim(Array))==4){
    if(verbose){print('Assume colmajor, detect 4 dimensional input')}
    N = dim(Array)[4]
    first_dim = dim(Array)[1:3]
  }else{
    if(verbose){print('Assume colmajor, detect 2 dimensional input')}
    N = dim(Array)[2]
    first_dim = dim(Array)[1]
  }
  n.pass = - (-N %/% max.compute.size)
  rest = N %% max.compute.size
  
  # We compute predictions per data slice
  for(i in 1:n.pass){
    not.complete = rest>0 & i==n.pass
    # We provide the input data to the executor
    if(not.complete){band=(N-rest+1):N}else{band = (i-1)*max.compute.size + 1:max.compute.size}
    
    # We create an executor for computing predictions on the data slice
    prediction = mx.simple.bind(predictor.symbol,
                                ctx=devices,
                                data= c(first_dim,length(band)),
                                label = c(dim(y)[1],length(band)))
    # arguments parameters are the model weights
    if(length(dim(Array))==4){
      mx.exec.update.arg.arrays(prediction,c(list(data=mx.nd.array(Array[,,,band,drop=F]),label=mx.nd.array(y[,band,drop=F])), params),match.name=T)
    }else{
      mx.exec.update.arg.arrays(prediction,c(list(data=mx.nd.array(Array[,band,drop=F]),label=mx.nd.array(y[,band,drop=F])),params),match.name=T)
    }
    # auxiliary parameters are hyper-parameters for which gradient is not computed 
    # They may be used as global statistics, especially for adaptative SGD algorithms 
    if(!is.null(aux.params)){mx.exec.update.aux.arrays(prediction,aux.params,match.name=T)}
    
    # is.train=F must be specified for prediction
    # Batchnorm is then applied with the current global statistics of aux.params
    # and those aux.params are unaffected by the current mini-batch
    mx.exec.forward(prediction,is.train=F)
    if(i==1){
      # first loss slice
      lossVec = as.array(prediction$outputs[[1]])
    }else{
      # concatenate to previous slices
      if(is.null(dim(lossVec))){
        lossVec = c(lossVec, as.array(prediction$outputs[[1]]) )
      }else{
        if(i==2 & verbose){print('detect multilabel, returns prediction matrix')}
        lossVec = cbind( lossVec , as.array(prediction$outputs[[1]]) )
      }
    }
    gc(reset=T)
  }
  return(lossVec)
}

top.K.accuracy = function(pred,label.matrix,K){
  score = 0
  for(i in 1:dim(label.matrix)[2]){
    score = score + which(label.matrix[,i]==1)%in%order(pred[,i],decreasing = T)[1:K]
  }
  return(score/dim(label.matrix)[2])
}


plot.RGBarray = function(RGBarray){
  tmp= array(as.vector(RGBarray),dim=c(32,32,1,3))
  im = cimg(tmp)
  plot(im)
}

#####
# Use MxNet Nd Arrays
#####

## Simple vector
x <- 1:3
mat <- mx.nd.array(x)

mat <- mat + 1.0
mat <- mat + mat
mat <- mat - 5
mat <- 10 / mat
mat <- 7 * mat
mat <- 1 - mat + (2 * mat) / (mat + 0.5)
as.array(mat)

## Matrix
x <- as.array(matrix(1:4, 2, 2))

mx.ctx.default(mx.cpu(1))
print(mx.ctx.default())
print(is.mx.context(mx.cpu()))

mat <- mx.nd.array(x)
mat <- (mat * 3 + 5) / 10
as.array(mat)

## Matrix GPU
x <- as.array(matrix(1:4, 2, 2))

mx.ctx.default(mx.gpu())
print(mx.ctx.default())
print(is.mx.context(mx.cpu()))

mat <- mx.nd.array(x)
mat2 <- (mat * 3 + 5) / 10
as.array(mat2)

## Matrices dot product Mxnet-GPU vs base-R
n = 1000
a <- mx.rnorm(c(n, n),ctx=mx.gpu()) # create a 2-by-3 matrix on cpu
b <- mx.rnorm(c(n, n), ctx=mx.gpu()) # create a 2-by-3 matrix on cpu

deb = Sys.time()
c = mx.nd.dot(a,b)
print(Sys.time()-deb)

aa = as.array(a)
bb = as.array(b)

deb = Sys.time()
cc = aa %*% bb
print(Sys.time()-deb)

rm(list = ls())
gc(reset=T)



#####
# NB: SET THE SEED
#####

# Use mx.set.seed instead of set.seed (doesn't work with mxnet number generator)
# the seed is device specific=> generated numbers will not be the same between 
# GPU and CPU
cont = mx.gpu()
mx.set.seed(0)
mx.runif(c(1,4),0,1,ctx=cont)

cont = mx.cpu(0)
mx.set.seed(0)
mx.runif(c(1,4),0,1,ctx=cont)


#####
# Test executor 
#####

ve = array(c(1:3,3:1),dim=c(3,2))
weights = matrix(c(1,0,0,0,1,0,0,0,1),3,3)
# TEST executor
data = mx.symbol.Variable(name="data")
linPred = mx.symbol.FullyConnected( data = data, num_hidden = 3 , name="linpred" )
output = mx.symbol.softmax(linPred,axis=1,name="SoftMax")

Executor = mx.simple.bind(output,
                          ctx=mx.cpu(),
                          data= dim(ve))

# on lui donne les donn�es d'entr�e (et de sortie pour la loss)
# arguments parameters
mx.exec.update.arg.arrays(Executor, list(data=mx.nd.array(ve),linpred_weight=mx.nd.array(weights)) , match.name=T)

# is.train=F very important, batchnorm is calculated with global statistics of aux.params
# rather than with current batch statistics
mx.exec.forward(Executor,is.train=F)

p = as.array(Executor$outputs[[1]])
print(p)



#####
# CIFAR-100
#####

dir = "C:/Users/Christophe/pCloud local/0_These/Autres formes de travaux/19_09_03 Seminaire R MXNET amap/"

### LOAD TRAIN
n = 50000
labels = rep(NA,n)
coarse = rep(NA,n)

gc(reset=T)
to.read = file(paste(dir,"cifar-100-binary/train.bin",sep=""), "rb")
v = readBin(to.read, "integer",n=n*3074, size=1 , signed=F)

# Extract images in an array
train.array = array(0,dim = c(32,32,3,n))
for(i in 1:n){
  TMP = NULL
  labels[i] = v[(i-1)*3074 +2 ]
  coarse[i] = v[(i-1)*3074 + 1]
  for(ch in 1:3){
    tmp = v[(i-1)*3074 + 2 + (1:1024) + (ch - 1 )*1024]
    Mtmp= as.vector(matrix(tmp,32,32))
    TMP=c(TMP,Mtmp)
    train.array[,,,i] = array(TMP , dim=c(32,32,3))
  }
}

#tmp = array(as.vector(arrays[[1]]),dim=c(32,32,1,3))
#im = cimg(tmp)
#plot(im)
rm(to.read)

plot.RGBarray(train.array[,,,1])

# Make TrAIN label array

Y.train = array(0,dim=c(length(unique(labels)),length(labels)))
for(i in 1:length(labels)){
  Y.train[labels[i]+1,i] = 1 
}

### LOAD TEST

n = 10000
test.coarse.labels = rep(NA,n)
gc(reset=T)
to.read = file(paste(dir,"cifar-100-binary/test.bin",sep=""), "rb")
v = readBin(to.read, "integer",n=n*3074, size=1 , signed=F)
# Extract images in an array
test.array = array(0,dim = c(32,32,3,n))
for(i in 1:n){
  TMP = NULL
  test.coarse.labels[i] = v[(i-1)*3074 + 1]
  for(ch in 1:3){
    tmp = v[(i-1)*3074 + 2 + (1:1024) + (ch - 1 )*1024]
    Mtmp= as.vector(matrix(tmp,32,32))
    TMP=c(TMP,Mtmp)
    test.array[,,,i] = array(TMP , dim=c(32,32,3))
  }
}

# make TEST labels array
Y.test = array(0,dim=c(length(unique(test.coarse.labels)),length(test.coarse.labels)))
for(i in 1:length(labels)){
  Y.test[test.coarse.labels[i]+1,i] = 1 
}

### LABEL NAMES
# Fine label name 
setwd(paste(dir,'cifar-100-binary/',sep=""))
labTab = read.csv('fine_label_names.txt',sep="",header=F)
colnames(labTab)[1] = "name"
labTab$id = 0:99

nameLab = function(ids,tab=labTab){tmp = merge(data.frame(id=ids),tab,by="id",all.x=T,sort=F); return(as.character(tmp$name)) }

nameLab(labels[1:10])

# Coarse label name 
setwd(paste(dir,'cifar-100-binary/',sep=""))
labTab = read.csv('coarse_label_names.txt',sep="",header=F)
colnames(labTab)[1] = "name"
labTab$id = 0:19

nameLab = function(ids,tab=labTab){tmp = merge(data.frame(id=ids),tab,by="id",all.x=T,sort=F); return(as.character(tmp$name)) }

nameLab(coarse[1:10])

labels = coarse

#####
# Design CNN
#####

n_labels = dim(Y.train)[1]

data = mx.symbol.Variable(name = "data")
label = mx.symbol.Variable(name="label")
# 1st convolutional layer
conv1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), 
                              pad = c(2, 2),
                              num_filter = 64,
                              name = "conv1")
relu1 = mx.symbol.Activation(data = conv1,
                             act_type = "relu",
                             name = "relu1")
pool1 = mx.symbol.Pooling(data = relu1,
                          pool_type = "max",
                          kernel = c(4,4),
                          stride=c(4,4),
                          name = "pool1")
bn1   = mx.symbol.BatchNorm(name="bn1",
                            data=pool1,
                            fix_gamma=F,
                            momentum = 0.9,
                            eps = 2e-5)
# 2nd convolutional layer
conv2 = mx.symbol.Convolution(data = bn1, 
                              kernel = c(5, 5),
                              pad = c(2, 2),
                              num_filter = 100, 
                              name = "conv2")
relu2 = mx.symbol.Activation(data = conv2, 
                             act_type = "relu", 
                             name = "relu2")
pool2 = mx.symbol.Pooling(data = relu2, 
                          pool_type = "max",
                          kernel = c(8, 8),
                          stride =c(8, 8), 
                          name = "pool2")

# Flattening
flatten = mx.symbol.Flatten(data = pool2, 
                            name = "flatten")
bn2   = mx.symbol.BatchNorm( name="bn2",
                             data=flatten,
                             fix_gamma=F,
                             momentum=0.9 ,
                             eps= 2e-5)

# Fully connected layer
fc1 = mx.symbol.FullyConnected(data = bn2,
                               num_hidden = 300,
                               name = "fc1")
rel3 = mx.symbol.Activation(data = fc1,
                            act_type = "relu",
                            name = "relu3")
bn4   = mx.symbol.BatchNorm( name="bn4",
                             data=rel3,
                             fix_gamma=F,
                             momentum=0.9 ,
                             eps= 2e-5)

# Linear Predictor
out = mx.symbol.FullyConnected(data = bn4, num_hidden = n_labels, name = "out")

# log-Loss
QuickLoss = mx.symbol.SoftmaxOutput(data = out , label=label , multi.output = T, name="cross_entropy")

modelSymbols = list(loss=QuickLoss)

# Visualize model
graph = graph.viz(out,type="graph",direction="LR")
print(graph)

#####
# Design customized loss
#####

softmax = mx.symbol.softmax(data= out , axis=1,  name="softmax")

loss= mx.symbol.MakeLoss(data= 0  - label * mx.symbol.log( softmax ) , name="cross_entropy_custom")

modelSymbols = list(loss=loss,pred=softmax)

#####
# SubSample train set and make Validation set
#####

# Subsample data and make validaion set
bag = 1:length(labels)
sampled = sample(bag,49000)
bag = setdiff(bag,sampled)

Xtrain =train.array[,,,sampled,drop=F]
labtrain = Y.train[,sampled,drop=F]

validSelec = sample(bag,1000)
Xvalid =train.array[,,,validSelec,drop=F]
Yvalid = Y.train[,validSelec,drop=F]

#####
# Learning with SGD Momentum
#####

# Training parameters
n_it= 500
batch.size = 32
saveDir = dir
devices = mx.gpu()
lr = rep(5e-6,n_it+1)
modelName = "finish2"
loadModel = F

if(F){
  mx.set.seed(2019)
  model= mx.model.FeedForward.create(symbol=modelSymbols$loss[[1]],
                                     X=Xtrain,y=labtrain,                
                                     ctx=mx.cpu(),begin.round=1,num.round = 1,
                                     learning.rate=0.,
                                     initializer = mx.init.uniform(0.03),
                                     array.layout = "colmajor")
  Names = names(model$arg.params)
  E_G = lapply(Names,function(arg) 0.*as.array(model$arg.params[arg][[1]]) )
  names(E_G)=Names
}else{
  # ADD MODEL
  model = NULL
}

tab = data.frame( it = 0:n_it,tr.loss = NA)
valid = data.frame( it = 0:n_it,loss = NA)
# prediction sample 
N = dim(labtrain)[2]
samplo = sample(1:N,500)
priorLoss = - log(1/ dim(labtrain)[1]) /dim(labtrain)[1]
n.batch = -( -N %/% batch.size)
it=0
while (it<=n_it){
  print(paste('it :',it))
  
  # PREDICT on train
  p = predict.executor(modelSymbols$pred[[1]],
                       model$arg.params,
                       Xtrain[,,,samplo,drop=F],
                       devices = mx.cpu(),
                       aux.params = model$aux.params,
                       max.compute.size = 200)
  # COMPUTE MEAN TRAIN LOSS
  loss = loss.executor(predictor.symbol = modelSymbols$loss[[1]],
                       params =model$arg.params,
                       Array = Xtrain[,,,samplo,drop=F],
                       y = labtrain[,samplo,drop=F], 
                       devices = mx.cpu(),
                       aux.params = model$aux.params,
                       max.compute.size = 200)
  if(sum(is.na(loss))>0){loss[is.na(loss)] = 0}
  if(sum(is.infinite(loss))>0){
    print('Infinite terms in the train loss')
    tab$tr.loss[tab$it==it]= NA
  }else{
    meanLoss = mean(as.vector(loss))
    print(paste('Mean train Loss :',meanLoss))
    tab$tr.loss[tab$it==it]= meanLoss
  }
  
  # PREDICT on validation
  p = predict.executor(modelSymbols$pred[[1]],
                       model$arg.params,
                       Xvalid,
                       devices = mx.cpu(),
                       aux.params = model$aux.params,
                       max.compute.size = 200)
  # COMPUTE MEAN VALIDATION LOSS
  loss = loss.executor(predictor.symbol = modelSymbols$loss[[1]],
                       params =model$arg.params,
                       Array = Xvalid,
                       y = Yvalid, 
                       devices = mx.cpu(),
                       aux.params = model$aux.params,
                       max.compute.size = 200)
  if(sum(is.na(loss))>0){loss[is.na(loss)] = 0}
  if(sum(is.infinite(loss))>0){
    print('Infinite terms in the validation loss')
    valid$loss[tab$it==it]= NA
  }else{
    meanLoss = mean(as.vector(loss))
    print(paste('Mean validation Loss :',meanLoss))
    valid$loss[tab$it==it]= meanLoss
  }
  
  # PLOT LOSS
  ypl = tab$tr.loss[tab$it<=it]
  validPl = valid$loss[valid$it<=it]
  ylim_r=range(c(ypl,0,priorLoss,validPl),na.rm = T)
  d=data.frame(its =c(0:it,0:it,0,it,0,it),
               val=c(ypl,validPl,0,0,priorLoss,priorLoss),
               curve= c(rep("train loss",length(ypl)),rep("validation loss",length(ypl)),"saturated loss","saturated loss","prior loss","prior loss"))
  pl=ggplot(d,aes(x=its,y=val,group=curve,colour=curve))+geom_point()+geom_line()+coord_cartesian(ylim = ylim_r) +ylab('Mean train loss')+xlab('Number of epochs')+theme_bw()
  print(pl)
  
  # Tirage des mini-batchs
  sampli = 1:N
  batchs.list = list()
  for(k in 1:n.batch){
    reduce = (length(sampli)<batch.size) * (batch.size-length(sampli))
    batchs.list[[k]] = sample(sampli , batch.size - reduce , replace=F)
    sampli = sampli[!(sampli%in% batchs.list[[k]]) ]
  }
  
  executor = mx.simple.bind(model$symbol,data=dim(Xtrain[,,,batchs.list[[1]],drop=F]),grad.req ="write",ctx=devices)

  # pass over each mini-batch
  for(k in 1:n.batch){
    elems = batchs.list[[k]]
    if(length(elems)<batch.size){
      executor = mx.simple.bind(model$symbol,data=dim(Xtrain[,,,elems,drop=F]),grad.req ="write",ctx=devices)
    }
    cat('\r  Process ...',100*k/n.batch,' %                               \r')
    flush.console()
    GradAndAux = get.all.gradients(model,Xtrain[,,,elems,drop=F],labtrain[,elems,drop=F],executor)

    for(arg in names(GradAndAux[[2]])){
      # Update each auxiliary parameter set by extracting them from GradAndAux
      # We copy those components to the CPU 
      # because that is where our model components are located
      model$aux.params[arg][[1]] = mx.nd.copyto(GradAndAux[[2]][arg][[1]],mx.cpu())
    }
    G= GradAndAux[[1]]
    for(arg in Names){
      # We compute parameters delta through R 
      # because I had a crash with Mxnet from an unknown reason ... 
      E_G[arg][[1]] = 0.9 * E_G[arg][[1]] + lr[it+1]* as.array(G[arg][[1]])
      # We compute parameters update through Mxnet
      model$arg.params[arg][[1]] = model$arg.params[arg][[1]] -  mx.nd.array( E_G[arg][[1]],ctx=mx.cpu())
    }
    gc(reset=T)
  }
  it=it+1
  setwd(saveDir)
  mx.model.save(model,modelName,iteration=it)
}

setwd(saveDir)
# tab$valid.loss = valid$loss
# write.table(tab,paste(modelName,"_losses.csv",sep=""),sep=";",row.names=F,col.names=T)


#####
# Evaluate model
#####

setwd(saveDir)
model = mx.model.load("finish2",iteration = 6)

# Validation accuracy
p.valid = predict.executor(modelSymbols$pred[[1]],
                     model$arg.params,
                     Xvalid,
                     devices = mx.cpu(),
                     aux.params = model$aux.params,
                     max.compute.size = 200)

accuracy = data.frame(K=1:dim(Yvalid)[1],topK_accuracy=NA)
accuracy$topK_accuracy = sapply(accuracy$K,function(k) top.K.accuracy(pred=p.valid,label.matrix=Yvalid,K=k))
ggplot(accuracy,aes(x=K,y=topK_accuracy))+geom_point()+geom_line()+theme_bw()+scale_y_continuous(limits = c(0,1))+ggtitle('Validation TopK-accuracy vs K')

# Test accuracy
p.test = predict.executor(modelSymbols$pred[[1]],
                           model$arg.params,
                           test.array,
                           devices = mx.cpu(),
                           aux.params = model$aux.params,
                           max.compute.size = 200)
accuracy = data.frame(K=1:dim(Y.test)[1],topK_accuracy=NA)
accuracy$topK_accuracy = sapply(accuracy$K,function(k) top.K.accuracy(pred=p.test,label.matrix=Y.test,K=k))
ggplot(accuracy,aes(x=K,y=topK_accuracy))+geom_point()+geom_line()+theme_bw()+scale_y_continuous(limits = c(0,1))+ggtitle('TEST TopK-accuracy vs K')

# Try on a single test data

i = sample(1:dim(test.array)[2],1)

idx = order(p.test[,i],decreasing = T)
print('Pr�dictions')
print(data.frame(rank=1:dim(p)[1],proba=p[idx,i],name=nameLab(idx-1,tab=labTab)))

plot.RGBarray(test.array[,,,i])


#####
# Learn a CNN with high level wrapper
#####

# Define a metric displayed at every epoch
log.loss.metric = function(label,pred){
  return(mx.nd.mean( - label * mx.nd.log(pred) ))
}

# Training parameters
n_it= 100
batch.size = 32
saveDir = dir
devices = mx.gpu()
lr = rep(1e-5,n_it+1)

# Learn the model
mx.set.seed(2019)
model <- mx.model.FeedForward.create(symbol=modelSymbols$loss[[1]],
                                     X=Xtrain,
                                     y=labtrain,
                                     ctx=devices,
                                     begin.round=1,
                                     num.round=10,
                                     array.batch.size=batch.size,
                                     learning.rate=lr[1],
                                     eval.data = list(data=Xvalid,label=Yvalid),
                                     momentum=0.9,  
                                     initializer = mx.init.uniform(0.03),
                                     eval.metric= mx.metric.custom(name = "logLoss",log.loss.metric),
                                     batch.end.callback=mx.callback.save.checkpoint("wrapper")
)


