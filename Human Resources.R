setwd("D:/AI & Data Science/R/Project 4 Human Resources")

hr_test=read.csv("hr_test.csv",stringsAsFactors = F)
hr_train=read.csv("hr_train.csv", stringsAsFactors = F)

library(dplyr)
library(car)

hr_test$left=NA

hr_test$data="test"
hr_train$data="train"

hr_all=rbind(hr_test,hr_train)

lapply(hr_all,function(x) sum(is.na(x)))

View(hr_all)
glimpse(hr_all)
#numeric-satisfaction_level,last_evaluation,number_project,average_montly_hours
        #time_spend_company,
#dummy-sales,salary
#boolean-Work_accident,promotion_last_5years,

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

sort(table(hr_all$salary))

hr_all=CreateDummies(hr_all,"sales",500)
hr_all=CreateDummies(hr_all,"salary",1000)

hr_train=hr_all %>% filter(data=='train') %>% select(-data)
hr_test=hr_all %>% filter(data=='test') %>% select(-data,-left)

set.seed(2)
s=sample(1:nrow(hr_train),.75*nrow(hr_train))
hr_train1=hr_train[s,]
hr_train2=hr_train[-s,]

library(pROC)

#Logistic Regression Model
for_vif=lm(left~.,data=hr_train1)

sort(vif(for_vif),decreasing = T)

for_vif=lm(left~.,data=hr_train1)

log_fit=glm(left~.,data=hr_train1)

log_fit
log_fit=step(log_fit)

summary(log_fit)

formula(log_fit)

val.score=predict(log_fit,newdata = hr_train2,type='response')

auc(roc(hr_train2$left,val.score))

#DTree Model
library(rpart)
library(rpart.plot)
library(tidyr)
require(rpart)
library(tree)

dtModel = tree(left~.,data=hr_train1)
plot(dtModel)
dtModel

val.score=predict(dtModel,newdata = hr_train2)
auc(roc(hr_train2$left,val.score))

#Random Forest Model
library(randomForest)
randomForestModel = randomForest(left~.,data=hr_train)
d=importance(randomForestModel)
d
d=as.data.frame(d)
names(d)
d$IncNodePurity=rownames(d)
d %>% arrange(desc(IncNodePurity))

val.score=predict(randomForestModel,newdata = hr_test)
auc(roc(hr_train2$left,val.score))

score=predict(randomForestModel,newdata= hr_test, type="response")

write.table(score,file ="Abhilash_Singh_P4_part2.csv",
            row.names = F,col.names="left")

#GBM Model
library(gbm)
library(cvTools)

param=list(interaction.depth=c(1:10),
           n.trees=c(700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  set.seed(2)
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=10
my_params=subset_paras(param,num_trials)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

myauc=0

for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  
  params=my_params[i,]
  
  k=cvTuning(gbm,left~.,data=hr_train,
             tuning =params,
             args = list(distribution="bernoulli"),
             folds = cvFolds(nrow(hr_train), K=10, type = "random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="response",n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this>myauc){
    print(params)
    
    myauc=score.this
    print(myauc)
    
    best_params=params
  }
  print('DONE')
}

myauc

best_params

best_params=data.frame(interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.01,
                       n.minobsnode=1)

product.gbm.final=gbm(left~.,data=hr_train,
                      n.trees = best_params$n.trees,
                      n.minobsinnode = best_params$n.minobsnode,
                      shrinkage = best_params$shrinkage,
                      interaction.depth = best_params$interaction.depth,
                      distribution = "bernoulli")

product.gbm.final

test.pred=predict(product.gbm.final,newdata=hr_test,
                  n.trees = best_params$n.trees,type="response")

#Cutoff
train.score=predict(product.gbm.final,newdata = hr_train,
                    n.trees = best_params$n.trees,type='response')

real=hr_train$left
cutoffs=seq(0.001,0.999,0.001)

cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  
  predicted=as.numeric(train.score>cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  
  KS=(TP/P)-(FP/N)
  
  #print(paste0('KS Score: ',KS))
  
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,
                    c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]

max(cutoff_data$KS)
1-(0.025/0.6886201)

View(cutoff_data)

#### visualise how these measures move across cutoffs
library(ggplot2)
ggplot(cutoff_data,aes(x=cutoff,y=M))+geom_line()

library(tidyr)

cutoff_long=cutoff_data %>% 
  gather(Measure,Value,Sn:M)

ggplot(cutoff_long,aes(x=cutoff,y=Value,color=Measure))+geom_line()

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff

# now that we have our cutoff we can convert score to hard classes

test.predicted=(test.pred>my_cutoff)
test.predicted
write.table(test.predicted,file ="Abhilash_Singh_P4_part2.csv",
            row.names = F,col.names="left")


############################Quiz############################################
#1
#linear regression

#2
table(hr_train$promotion_last_5years)

#3
s1=hr_train %>% 
  filter(left==0)

round(var(s1$satisfaction_level),4)

#4
library(ggplot2)
View(hr_train)
glimpse(hr_train)

p=ggplot(hr_train,aes(x=average_montly_hours,y=left))
p
p+geom_line()
p+geom_point()
p+geom_point()+geom_line()+geom_smooth()

ggplot(hr_train, aes(x=average_montly_hours)) + 
  geom_histogram(binwidth=.25, colour="black", fill="white")

ggplot(hr_train,aes(x=average_montly_hours))+geom_density(color="red")+
  geom_histogram(aes(y=..density..,alpha=0.5))+
  stat_function(fun=dnorm,aes(x=average_montly_hours),color="green")

#5
table(hr_train$salary)
hr_train %>%
  select(salary,left) %>% 
  group_by(salary) %>% 
  summarise(sum_left=sum(left==1))

table(hr_train$left)

#6
round(cor(hr_train$last_evaluation, hr_train$average_montly_hours),2)
cor.test(hr_train$last_evaluation, hr_train$average_montly_hours)

#7
table(hr_train$Work_accident)
glimpse(hr_train)

s2=hr_train %>% 
  select(left,Work_accident) %>% 
  filter(Work_accident==1)
table(s2$left)
round(270/1515,2)

#8
s3=hr_train %>%
  filter(left==1) %>%
  select(time_spend_company)
median(s3$time_spend_company)
  
#9
s4=hr_train %>% 
  select(sales,average_montly_hours) %>% 
  group_by(sales) %>% 
  summarise(med=median(average_montly_hours))
max(s4$med)
View(s4)

#10
s5=hr_train %>% 
  select(left,number_project) %>% 
  group_by(left) %>% 
  summarise(projects=sum(number_project))

View(s5)
############################################################################