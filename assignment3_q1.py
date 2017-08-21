import gzip
from collections import defaultdict
import numpy
import urllib
import scipy.optimize
import random
import math
from sklearn import svm
import sklearn

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

    
training_rate=[]
training_set=[]
allHelpful_train = []
userHelpful_train = defaultdict(list)
itemHelpful_train = defaultdict(list)
item_rating=defaultdict(list)
allHelpful_validate = []
userHelpful_validate = defaultdict(list)
validate_set=[]
training_year=[]
training_time=[]

count=1
for l in readGz("train.json.gz"):
    user,item = l['reviewerID'],l['itemID']
    if(count>=1 and count<=19999):
        training_set.append(l);
        allHelpful_train.append(l['helpful'])
        if(l['helpful']['outOf']==0):
            training_rate.append(0)
        else:
            training_rate.append(l['helpful']['nHelpful']/l['helpful']['outOf'])
        userHelpful_train[user].append(l['helpful'])
        itemHelpful_train[item].append(l['helpful'])
        temp=l['reviewTime'].split()
        training_year.append(int(temp[2]))
        training_time.append(l['unixReviewTime'])
        item_rating[item].append(l['rating'])
        count=count+1;
    elif(count>19999 and count<=20000):
        validate_set.append(l);
        allHelpful_validate.append(l['helpful'])
        userHelpful_validate[user].append(l['helpful'])
        count=count+1;


averageRate = sum([x['nHelpful'] for x in allHelpful_train]) * 1.0 / sum([x['outOf'] for x in allHelpful_train])
alpha=averageRate
print(averageRate)

avg_itemRating = {}
for i in item_rating:
  avg_itemRating[i]=numpy.sum(item_rating[i])/len(item_rating[i])

userRate = {}
for u in userHelpful_train:
  totalU = sum([x['outOf'] for x in userHelpful_train[u]])
  if totalU > 0:
    userRate[u] = sum([x['nHelpful'] for x in userHelpful_train[u]]) * 1.0 / totalU
  else:
    userRate[u] = 0

itemRate = {}
for i in itemHelpful_train:
  totalU = sum([x['outOf'] for x in itemHelpful_train[i]])
  if totalU > 0:
    itemRate[i] = sum([x['nHelpful'] for x in itemHelpful_train[i]]) * 1.0 / totalU
  else:
    itemRate[i] = 0

min_year=min(training_year);
min_unixReviewTime=min(training_time)
print(min_unixReviewTime)
def feature(datum):
  feat = [1]
  review=datum['reviewText']
  words=review.split()
  feat.append(len(words))
  feat.append(datum['rating'])
  temp=datum['reviewTime'].split()
  feat.append(int(temp[2])-min_year+1)
  feat.append(datum['helpful']['outOf'])
  summary=datum['summary']
  s_words=summary.split()
  feat.append(len(s_words))
  y=[len(datum['categories'][i]) for i in range(len(datum['categories']))]
  feat.append(numpy.sum(y))
  feat.append(datum['categoryID'])
  #feat.append(datum['unixReviewTime']-min_unixReviewTime)
##  if datum['reviewerID'] in userRate:
##    feat.append(userRate[datum['reviewerID']]-averageRate)
##  else:
##    feat.append(0.5)
##  if datum['itemID'] in itemRate:
##    if datum['reviewerID'] in userRate:
##     feat.append(itemRate[datum['itemID']]*userRate[datum['reviewerID']])
##  else:
##    feat.append(0)               
  return feat


def sigmoid(gamma):  
  if gamma < 0:
      return 1 - 1 / (1 + math.exp(gamma))
  return 1 / (1 + math.exp(-gamma))


def inner(x,y):
  temp=[(x[i]*y[i]) for i in range(len(x))]
  temp=numpy.array(temp)
  temp=numpy.sum(temp)
  return temp

def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= math.log(1 + math.exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  return -loglikelihood


# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  for k in range(len(theta)):
        dl[k]=dl[k]-(2*lam*theta[k]) 
        for i in range(len(X)):
                # Fill in code for the derivative
            logit = inner(X[i], theta)
            
            dl[k]=dl[k]+ ((1-sigmoid(logit))*X[i][k])
            if not y[i]:
                dl[k]=dl[k]-X[i][k]
  return numpy.array([-x for x in dl])                    
  # Negate the return value since we're doing gradient *ascent*

  
  
X_features_for_Regressor=[feature(d) for d in training_set]
X_features_for_Regressor=numpy.array(X_features_for_Regressor)

y_train= [d>=0.5 for d in training_rate]


  

##initial_guess=[];
##initial_guess.append(averageRate)
##for i in range(1,len(X_features_for_Regressor[0])):
##  initial_guess.append(random.random())
##  
##
##for lam in [0.01]:
##  
##  theta,l,info = scipy.optimize.fmin_l_bfgs_b(f,initial_guess, fprime, args = (X_features_for_Regressor,y_train,lam))
##  sum1=0;
##
##
##  for u in validate_set:
##      review=u['reviewText']
##      words=review.split()
##      temp=u['reviewTime'].split()
##      summary=u['summary']
##      s_words=summary.split()
##      y=[len(u['categories'][i]) for i in range(len(u['categories']))]
##      sum1=sum1+ abs(u['helpful']['nHelpful']-((sigmoid(theta[0]+(theta[1]*len(words))+(theta[2]*u['rating'])+(theta[3]*(int(temp[2])-min_year+1))+(theta[4]*u['helpful']['outOf'])+(theta[5]*len(s_words))))*u['helpful']['outOf']))
##      
##           
##  mse_three=sum1/len(validate_set)
##  print(mse_three,'lam',lam,'logistic')

  
sum1=0
lr=sklearn.linear_model.LogisticRegression()
y_train=numpy.array(y_train)
lr.fit(X_features_for_Regressor,y_train)

for u in validate_set:
  
  review=u['reviewText']
  words=review.split()
  temp=u['reviewTime'].split()
  summary=u['summary']
  s_words=summary.split()
  y=[len(u['categories'][i]) for i in range(len(u['categories']))]
  if u['reviewerID'] in userRate:
    avg_u=userRate[u['reviewerID']]
  else:
    avg_u=0
  if u['itemID'] in itemRate:
    avg_i=itemRate[u['itemID']]
  else:
    avg_i=0
  temp=lr.decision_function([[1,len(words),u['rating'],(int(temp[2])-min_year+1),u['helpful']['outOf'],len(s_words),numpy.sum(y),u['categoryID']]])
  if u['itemID'] in itemRate and avg_itemRating[u['itemID']]>=3:    
    temp=1*u['helpful']['outOf']
  elif avg_u>=averageRate:
    temp=1*u['helpful']['outOf']
  elif avg_i>=averageRate:
    temp=1*u['helpful']['outOf']
  else:
    temp=(sigmoid(temp[0])*u['helpful']['outOf'])
  sum1=sum1+ abs(u['helpful']['nHelpful']-temp)
      
mse_three=sum1/len(validate_set)
print(mse_three,'class_logistic')


##clf = sklearn.svm.SVR()
##training_rate=numpy.array(training_rate)
##clf.fit(X_features_for_Regressor,training_rate)
##sum1=0;
##
##for u in validate_set:
##  
##  review=u['reviewText']
##  words=review.split()
##  temp=u['reviewTime'].split()
##  summary=u['summary']
##  s_words=summary.split()
##  y=[len(u['categories'][i]) for i in range(len(u['categories']))]
##  if u['reviewerID'] in userRate:
##    avg_u=userRate[u['reviewerID']]
##  else:
##    avg_u=0.5
##  if u['itemID'] in itemRate:
##    avg_i=itemRate[u['itemID']]
##  else:
##    avg_i=0.5
##  temp=clf.predict([[1,len(words),u['rating'],(int(temp[2])-min_year+1),u['helpful']['outOf'],len(s_words),numpy.sum(y),u['categoryID']]])
##  if u['itemID'] in itemRate and avg_itemRating[u['itemID']]>=3:    
##    temp=1*u['helpful']['outOf']
##  elif avg_u>=averageRate:
##    temp=1*u['helpful']['outOf']
##  elif avg_i>=averageRate:
##    temp=1*u['helpful']['outOf']
##  else:  
##    temp=(temp[0]*u['helpful']['outOf'])
##  sum1=sum1+ abs(u['helpful']['nHelpful']-temp)
##
##
##mse_three=sum1/len(validate_set)
##print(mse_three,'SVM')








##theta,residuals,rank,s = numpy.linalg.lstsq(X_features_for_Regressor, training_rate)
##print('nor',theta)
##sum1=0;
##
##for u in validate_set:
##  review=u['reviewText']
##  words=review.split()
##  temp=u['reviewTime'].split()
##  summary=u['summary']
##  s_words=summary.split()
##  y=[len(u['categories'][i]) for i in range(len(u['categories']))]
##  if u['reviewerID'] in userRate:
##    avg_u=userRate[u['reviewerID']]
##  else:
##    avg_u=0
##  if u['itemID'] in itemRate:
##    avg_i=itemRate[u['itemID']]
##  else:
##    avg_i=0
##  if u['itemID'] in itemRate and avg_itemRating[u['itemID']]>=3:    
##    temp=1*u['helpful']['outOf']
##  elif avg_u>=averageRate:
##    temp=1*u['helpful']['outOf']
##  elif avg_i>=averageRate:
##    temp=1*u['helpful']['outOf']  
##  else:
##    temp=(theta[0]+(theta[1]*len(words))+(theta[2]*u['rating'])+(theta[3]*(int(temp[2])-min_year+1))+theta[4]*u['helpful']['outOf']+theta[5]*len(s_words)+theta[6]*numpy.sum(y)+theta[7]*u['categoryID']+theta[8]*avg_i*avg_u)*u['helpful']['outOf']
##  sum1=sum1+ abs(u['helpful']['nHelpful']-temp)
##    
##         
##mse_three=sum1/len(validate_set)
##print('nor',mse_three)


predictions = open("predictions_Helpful_ass_temp_helpf5.txt", 'w')
predictions.write("userID-itemID-outOf,prediction\n")
for u in readGz("test_Helpful.json.gz"):
  review=u['reviewText']
  words=review.split()
  temp=u['reviewTime'].split()
  summary=u['summary']
  s_words=summary.split()
  y=[len(u['categories'][i]) for i in range(len(u['categories']))]
  if u['reviewerID'] in userRate:
    avg_u=userRate[u['reviewerID']]
  else:
    avg_u=0
  if u['itemID'] in itemRate:
    avg_i=itemRate[u['itemID']]
  else:
    avg_i=0
  temp=lr.decision_function([[1,len(words),u['rating'],(int(temp[2])-min_year+1),u['helpful']['outOf'],len(s_words),numpy.sum(y),u['categoryID']]])
  if u['itemID'] in itemRate and avg_itemRating[u['itemID']]>=3:    
    temp=1*u['helpful']['outOf']
  elif avg_u>=averageRate:
    temp=1*u['helpful']['outOf']
  elif avg_i>=averageRate:
    temp=1*u['helpful']['outOf']  
  else:
    temp=(sigmoid(temp[0])*u['helpful']['outOf'])
  predictions.write(u['reviewerID'] + '-' + u['itemID'] + '-' + str(u['helpful']['outOf']) + ',' + str(temp) + '\n')
      
predictions.close()

    


        
        
        
        
  
  
  


