import gzip
from collections import defaultdict
import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


test_set=[]
training_set=[]
allRatings_train = []
userRatings_train = defaultdict(list)
user_item_train=defaultdict(list)
item_user_train=defaultdict(list)
validate_set=[]
itemRatings_train = defaultdict(list)
count=1
for l in readGz("train.json.gz"):
    user,item = l['reviewerID'],l['itemID']
    if(count>=1 and count<=10000):
        training_set.append(l);
        allRatings_train.append(l['rating'])
        userRatings_train[user].append(l['rating'])
        itemRatings_train[item].append(l['rating'])
        user_item_train[user].append(l['itemID'])
        item_user_train[item].append(l['reviewerID'])
        count=count+1;
    elif(count>10000 and count<=20000):
        validate_set.append(l);
        count=count+1;

globalAverage = sum(allRatings_train) / len(allRatings_train)

K=4
alpha=globalAverage

def sigmoid(x):
  return 1.0 / (1 + math.exp(-x))

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

beta_u={}
beta_i={}
gamma_u=defaultdict(list)
gamma_i=defaultdict(list)

init_theta=[];
init_theta.append(alpha)


for u in userRatings_train:
  beta_u[u]=random.random()
  init_theta.append(beta_u[u])
for i in itemRatings_train:
  beta_i[i]=random.random()
  init_theta.append(beta_i[i])

for u in userRatings_train:
  for k in range(0, K):
    gamma_u[u].append(random.random()*0.1)
    init_theta.append(gamma_u[u][k])

for i in itemRatings_train:
  for k in range(0, K):
    gamma_i[i].append(random.random()*0.1)
    init_theta.append(gamma_i[i][k])

lam=5    
    
def f(theta):
  alpha=theta[0];
  temp_beta_u=theta[1:1+len(userRatings_train)]
  temp_beta_i=theta[1+len(userRatings_train):1+len(userRatings_train)+len(itemRatings_train)]
  temp_gamma_u=theta[1+len(userRatings_train)+len(itemRatings_train):1+len(userRatings_train)+len(itemRatings_train)+(len(userRatings_train)*K)]
  temp_gamma_i=theta[1+len(userRatings_train)+len(itemRatings_train)+(len(userRatings_train)*K):1+len(userRatings_train)+len(itemRatings_train)+(len(userRatings_train)*K)+(len(itemRatings_train)*K)]

  beta_u={}
  beta_i={}
  gamma_u=defaultdict(list)
  gamma_i=defaultdict(list)

  ind=0;

  for u in userRatings_train:
    beta_u[u]=temp_beta_u[ind]
    ind=ind+1;
    
  ind=0;  
    
  for i in itemRatings_train:
    beta_i[i]=temp_beta_i[ind]
    ind=ind+1

  ind=0;      

  for u in userRatings_train:
    for k in range(0,K):
      gamma_u[u].append(temp_gamma_u[ind])
      ind=ind+1;
      
  ind=0;   
  for i in itemRatings_train:
    for k in range(0,K):
      gamma_i[i].append(temp_gamma_i[ind])
      ind=ind+1;
      
  obj=0;
  for l in training_set:
    obj=obj+(alpha+beta_u[l['reviewerID']]+beta_i[l['itemID']]+inner(gamma_u[l['reviewerID']],gamma_i[l['itemID']])-l['rating'])**2
  obj=obj+(lam*(sum((theta)**2)-alpha**2))
  return obj

def fprime(theta):
  alpha=theta[0];
  temp_beta_u=theta[1:1+len(userRatings_train)]
  temp_beta_i=theta[1+len(userRatings_train):1+len(userRatings_train)+len(itemRatings_train)]
  temp_gamma_u=theta[1+len(userRatings_train)+len(itemRatings_train):1+len(userRatings_train)+len(itemRatings_train)+(len(userRatings_train)*K)]
  temp_gamma_i=theta[1+len(userRatings_train)+len(itemRatings_train)+(len(userRatings_train)*K):1+len(userRatings_train)+len(itemRatings_train)+(len(userRatings_train)*K)+(len(itemRatings_train)*K)]
  beta_u={}
  beta_i={}
  gamma_u=defaultdict(list)
  gamma_i=defaultdict(list)
  ind=0;
  for u in userRatings_train:
    beta_u[u]=temp_beta_u[ind]
    ind=ind+1;    
  ind=0;      
  for i in itemRatings_train:
    beta_i[i]=temp_beta_i[ind]
    ind=ind+1
  ind=0;      
  for u in userRatings_train:
    for k in range(0,K):
      gamma_u[u].append(temp_gamma_u[ind])
      ind=ind+1;
  ind=0;   

  for i in itemRatings_train:
    for k in range(0,K):
      gamma_i[i].append(temp_gamma_i[ind])
      ind=ind+1;
  ind=0;    
  theta_der=[];
  der_alpha=0;
  for l in training_set:
     der_alpha=der_alpha+(2*(alpha+beta_u[l['reviewerID']]+beta_i[l['itemID']]+inner(gamma_u[l['reviewerID']],gamma_i[l['itemID']])-l['rating']))
  theta_der.append(der_alpha)
  for u in userRatings_train:
    temp=2*((sum([(alpha+beta_i[I]+beta_u[u]+inner(gamma_u[u],gamma_i[I])) for I in user_item_train[u]]))-(sum(userRatings_train[u])))+(2*lam*beta_u[u])
    theta_der.append(temp)
  for i in itemRatings_train:
    temp=2*((sum([(alpha+beta_i[i]+beta_u[U]+inner(gamma_u[U],gamma_i[i])) for U in item_user_train[i]]))-(sum(itemRatings_train[i])))+(2*lam*beta_i[i])
    theta_der.append(temp)
  for u in userRatings_train:
    for k in range(0, K):
     temp=2*((sum([(gamma_i[I][k]*(alpha+beta_i[I]+beta_u[u]+inner(gamma_u[u],gamma_i[I]))) for I in user_item_train[u]]))-(sum([(userRatings_train[u][j]*gamma_i[user_item_train[u][j]][k])for j in range(len(userRatings_train[u]))])))+(2*lam*gamma_u[u][k])
     theta_der.append(temp)
     
  for i in itemRatings_train:
    for k in range(0,K):
      temp=2*((sum([(gamma_u[U][k]*(alpha+beta_i[i]+beta_u[U]+inner(gamma_u[U],gamma_i[i]))) for U in item_user_train[i]]))-(sum([(itemRatings_train[i][j]*gamma_u[item_user_train[i][j]][k])for j in range(len(itemRatings_train[i]))])))+(2*lam*gamma_i[i][k])
      theta_der.append(temp);

  return numpy.array(theta_der)



init_theta=numpy.array(init_theta)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f,init_theta, fprime)

alpha=theta[0];
temp_beta_u=theta[1:1+len(beta_u)]
temp_beta_i=theta[1+len(beta_u):1+len(beta_u)+len(beta_i)]
temp_gamma_u=theta[1+len(beta_u)+len(beta_i):1+len(beta_u)+len(beta_i)+(len(gamma_u)*K)]
temp_gamma_i=theta[1+len(beta_u)+len(beta_i)+(len(gamma_u)*K):1+len(beta_u)+len(beta_i)+(len(gamma_u)*K)+(len(gamma_i)*K)]

beta_u={}
beta_i={}
gamma_u=defaultdict(list)
gamma_i=defaultdict(list)

ind=0;

for u in userRatings_train:
  beta_u[u]=temp_beta_u[ind]
  ind=ind+1;
  
ind=0;  
  
for i in itemRatings_train:
  beta_i[i]=temp_beta_i[ind]
  ind=ind+1;

ind=0;  
  

for u in userRatings_train:
  for k in range(0,K):
    gamma_u[u].append(temp_gamma_u[ind])
    ind=ind+1;

ind=0;   

for i in itemRatings_train:
  for k in range(0,K):
    gamma_i[i].append(temp_gamma_i[ind])
    ind=ind+1

  

      

####for lam in [10]:
####    
####    beta_u={}
####    beta_i={}
####
####    for u in userRatings_train:
####        beta_u[u]=random.random()
####    for i in itemRatings_train:
####        beta_i[i]=random.random()
####
####    alpha=globalAverage
####
####    convergence=0
####    
####    while(convergence==0):
####        prev_alpha=alpha
####        prev_beta_u=beta_u
####        prev_beta_i=beta_i
####
####        temp=0
####        for d in training_set:
####            
####            user,item = d['reviewerID'],d['itemID']
####            temp=temp+(d['rating']-(beta_u[user]+beta_i[item]))
####        alpha= temp/len(training_set)
####            
####        for u in userRatings_train:
####            beta_u[u]=((sum(userRatings_train[u]))- (sum([(alpha+beta_i[I]) for I in user_item_train[u]])))/(lam+len(user_item_train[u])) 
####        
####        for i in itemRatings_train:
####            beta_i[i]=((sum(itemRatings_train[i]))- (sum([(alpha+beta_u[U]) for U in item_user_train[i]])))/(lam+len(item_user_train[i]))
####
####        conv_beta_i=all(abs(beta_i[i]-prev_beta_i[i])< 10**-6 for i in itemRatings_train)
####        conv_beta_u=all(abs(beta_u[u]-prev_beta_u[u])< 10**-6 for u in userRatings_train)
####        conv_alpha=(abs(alpha-prev_alpha))<10**-6
####
####        conv=conv_beta_i and conv_beta_u and conv_alpha
####        
####        if(conv==True):
####            convergence=1
            


sum4=0;
  
    
for r in validate_set:
    if(r['itemID'] in itemRatings_train):
        if(r['reviewerID'] in userRatings_train):
            sum4=sum4+ (r['rating']-(int(alpha+beta_i[r['itemID']]+beta_u[r['reviewerID']]+inner(gamma_u[r['reviewerID']],gamma_i[r['itemID']]))))**2
        else:
            sum4=sum4+(r['rating']-(int(alpha+beta_i[r['itemID']])))**2
    elif(r['reviewerID'] in userRatings_train):
        sum4=sum4+ (r['rating']-(int(alpha+beta_u[r['reviewerID']])))**2
    else:
        sum4=sum4+ (r['rating']-(int(alpha)))**2

mse_modi=sum4/len(validate_set)
print("mse validate =",mse_modi,"for lambda =",lam)



predictions = open("predictions_Rating_ass_1.txt", 'w')
for l in open("pairs_Rating.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
    u,i = l.strip().split('-')
    if(i in itemRatings_train):
        if(u in userRatings_train):
            predict_rating=(alpha+beta_i[i]+beta_u[u]+inner(gamma_u[u],gamma_i[i]))
        else: 
            predict_rating=(alpha+beta_i[i])      
    elif(u in userRatings_train):
        predict_rating=(alpha+beta_u[u])
              
    else:
        predict_rating=(alpha) 
    predictions.write(u + '-' + i + ',' + str(predict_rating) + '\n')

 
predictions.close()    





        

    

    



