#!/usr/bin/env python
# coding: utf-8

# # Binary and Multiclass Classifiers:Perceptron and PA

# In[1]:


#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# read in the data, don't have to specify a file path because its already in the same directory
data=pd.read_csv('fashion-mnist_train.csv')
data1=pd.read_csv('fashion-mnist_test.csv')
#data.tail()
#data.head()


# In[3]:


# we create the training features by dropping the labels column, like so...
features=data.drop("label",axis=1) #for the training set 
features1=data1.drop("label",axis=1) #for the test set
features=np.array(features)
features1=np.array(features1)  #convert both into numpy array to ease the computation.


# In[4]:


#in the same fashion. we create the labels column form our initial data dataframes
labels=data['label'] #training set.....
labels=np.array(labels)
labels=np.where(labels%2==0,1,-1) #This here creates the binary class.
labels
####################################################################################################
labels1=data1['label']#test set
labels1=np.array(labels1)
labels1=np.where(labels1%2==0,1,-1)
labels1


# In[5]:


#Build the percpetron and PA weight updates.......... I will use a class:, this here is the perceptron updating.
class Binary:
    def __init__(self,tau=1,T=50):
        self.tau=tau
        self.T=T     # seen somewhere this being called epochs....
        self.weights=None
        
    def predict(self,feature_set):
        sum_=np.dot(feature_set,self.weight)
        return np.sign(sum_)
    def train(self, training_set_x,training_set_y):
        features=training_set_x.shape[1]
        error=[]
        accuracy=[]
        
        running_sum=0
        self.weight=np.zeros(features)
        
        for _ in range(self.T):
            sigma_error=0
            for i,j in enumerate(training_set_x):
                
                running_sum+=1
                y_hat=np.sign(np.dot(j,self.weight))
                
                if (y_hat==training_set_y[i]):
                    continue
                else:
                    self.weight=self.weight+(self.tau*training_set_y[i]*j)
                    sigma_error+=1
            error.append(sigma_error)
            accuracy.append(1-(sigma_error/running_sum))
      
        
        return(self.weight,accuracy,error, running_sum)


# In[6]:


#this is the passive aggressive algorithm

class Binary1:
    def __init__(self,tau=1,T=50):
        self.tau=tau
        self.T=T     # seen somewhere this being called epochs....
        self.weights=None
        
    def predict(self,feature_set):
        sum_=np.dot(feature_set,self.weight)
        return np.sign(sum_)
    def train(self, training_set_x,training_set_y):
        features=training_set_x.shape[1]
        error=[]
        accuracy=[]
        
        running_sum=0
        self.weight=np.zeros(features)
        
        for _ in range(self.T):
            sigma_error=0
            for i,j in enumerate(training_set_x):
                
                running_sum+=1
                y_hat=np.sign(np.dot(j,self.weight))
                
                if (y_hat==training_set_y[i]):
                    continue
                else:
                    self.tau=(1-(training_set_y[i]*np.dot(j,self.weight)))/np.square(np.linalg.norm(j)) #PA updating
                    self.weight=self.weight+(self.tau*training_set_y[i]*j)
                    sigma_error+=1
            error.append(sigma_error)
            accuracy.append(1-(sigma_error/running_sum))
        
        
        return(self.weight,accuracy,error, running_sum)


# In[7]:


#We now do the training for both the perceptron and PA.....
#PERCEPTRON
t=Binary()
r=t.train(features,labels)
weights,accuracy,errors,running_sum=r
iterations=list(range(0,50))
print(iterations)
#######################################################################
#PA
t1=Binary1()
features=np.array(features)
r1=t.train(features,labels)
weights1,accuracy1,errors1,running_sum1=r1
iterations1=list(range(0,50))
print(iterations1)


# ## 5.1a

# In[8]:


#Plotting the two update rules side by side........
plt.subplot(1, 2, 1)
plt.plot(iterations,errors)
plt.xlabel("iterations")
plt.ylabel("errors")
plt.title("Perceptron Updating")
plt.subplot(1, 2, 2)
plt.plot(iterations1,errors1)
plt.xlabel("iterations_PA")
plt.ylabel("errors_PA")
plt.title("PA Updating")


# #The curves look incredibly similar. the errors reduce with the number of training iterations. At approximately 10 iterations,
# #the curves "flatten out"- speaking in expectation terms. The spikes and the dips I will put down to some randomness in the data

# ## 5.1b

# In[9]:


#PA and Perceptron accuracy for both training and test data for 20 iterations
P=Binary(1,20)#perceptron updating
PA=Binary1(1,20)# PA updating.
features=np.array(features)
r_train=P.train(features,labels)#perceptron algorithm,training set
r_p_train=PA.train(features,labels)#PA ,training set
r_test=P.train(features1,labels1)#Perceptron algorithm,test set
r_p_test=PA.train(features1,labels1)#PA test set.....

#########################################################################################################
weights2,accuracy2,errors2,running_sum2=r_train
weights3,accuracy3,errors3,running_sum3=r_p_train
weights4,accuracy4,errors4,running_sum4=r_test
weights5,accuracy5,errors5,running_sum5=r_p_test


# In[10]:


#plot the graphs, side by side..........
iterations_=list(range(0,20))
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(iterations_, accuracy2)
# axs[0, 0].set_title('P_Training set')
# axs[0, 1].plot(iterations_,accuracy3 , 'tab:orange')
# axs[0, 1].set_title('PA_Training set')
# axs[1, 0].plot(iterations_,accuracy4, 'tab:green')
# axs[1, 0].set_title('P_Test_set')
# axs[1, 1].plot(iterations_, accuracy4, 'tab:red')
# axs[1, 1].set_title('PA_test_set')
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(iterations_, accuracy2)
axs[0, 0].set_title('Perceptron_train')
axs[0, 1].plot(iterations_, accuracy3, 'tab:orange')
axs[0, 1].set_title('PA_Train')
axs[1, 0].plot(iterations_, accuracy4, 'tab:green')
axs[1, 0].set_title('Perceptron test')
axs[1, 1].plot(iterations_, accuracy5, 'tab:red')
axs[1, 1].set_title('PA_TEST')

for ax in axs.flat:
    ax.set(xlabel='ITERATIONS', ylabel='ACCURACY')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


# ##The accuracy of both the PA and the perceptron updating for both sets of data increases almost in the same fashion attaining 
# #max trainig accuracy of ~ 1. they all look like the right half of a sigmoid curve.

# ## 5.1C

# In[12]:


#Average perceptron, perceptron updating
class Binary_Average:
    def __init__(self,tau=1,T=50):
        self.tau=tau
        self.T=T     # seen somewhere this being called epochs....
        self.weights=None
        
    def predict(self,feature_set):
        sum_=np.dot(feature_set,self.weight)
        return np.sign(sum_)
    def train(self, training_set_x,training_set_y):
        features=training_set_x.shape[1]
        error=[]
        accuracy=[]
        
        running_sum=0
        self.weight=np.zeros(features)
        
        for _ in range(self.T):
            sigma_error=0
            for i,j in enumerate(training_set_x):
                
                running_sum+=1
                y_hat=np.sign(np.dot(j,self.weight))
                
                if (y_hat==training_set_y[i]):
                    continue
                else:
                    self.weight=self.weight+(self.tau*training_set_y[i]*j)/training_set_x.shape[0] #weighted weights
                    sigma_error+=1
            error.append(sigma_error)
            accuracy.append(1-(sigma_error/running_sum))
        #print(running_sum)
        #print(error)
        #print(accuracy)
        
        return(self.weight,accuracy,error, running_sum)


# In[13]:


#average perceptron,PA updating
class Binary_Average1:
    def __init__(self,tau=1,T=50):
        self.tau=tau
        self.T=T     # seen somewhere this being called epochs....
        self.weights=None
        
    def predict(self,feature_set):
        sum_=np.dot(feature_set,self.weight)
        return np.sign(sum_)
    def train(self, training_set_x,training_set_y):
        features=training_set_x.shape[1]
        error=[]
        accuracy=[]
        
        running_sum=0
        self.weight=np.zeros(features)
        
        for _ in range(self.T):
            sigma_error=0
            for i,j in enumerate(training_set_x):
                
                running_sum+=1
                y_hat=np.sign(np.dot(j,self.weight))
                
                if (y_hat==training_set_y[i]):
                    continue
                else:
                    self.tau=(1-(training_set_y[i]*np.dot(j,self.weight)))/np.square(np.linalg.norm(j)) #PA updating
                    self.weight=self.weight+(self.tau*training_set_y[i]*j)/training_set_x.shape[0] #weighted weights...
                    sigma_error+=1
            error.append(sigma_error)
            accuracy.append(1-(sigma_error/running_sum))
      
        
        return(self.weight,accuracy,error, running_sum)


# In[14]:


#We now do the training for both the perceptron and PA.....
#PERCEPTRON
f=Binary_Average()
r_avg=f.train(features,labels)
weights_avg,accuracy_avg,errors_avg,running_sum_avg=r_avg
iterations=list(range(0,50))
#print(iterations)
#######################################################################
#PA
f1=Binary_Average1()
features=np.array(features)
r1_avg=f.train(features,labels)
weights1_avg,accuracy1_avg,errors1_avg,running_sum1_avg=r1_avg
iterations1=list(range(0,50))
#print(iterations1)


# In[15]:


P_avg=Binary_Average(1,20)
PA_avg=Binary_Average1(1,20)
features=np.array(features)
f_train=P_avg.train(features,labels)#perceptron algorithm,training set
f_p_train=PA_avg.train(features,labels)#PA ,training set
f_test=P_avg.train(features1,labels1)#Perceptron algorithm,test set
f_p_test=PA_avg.train(features1,labels1)#PA test set....

##################################################################################################
#Assign values
weights2_avg,accuracy2_avg,errors2_avg,running_sum2_avg=f_train
weights3_avg,accuracy3_avg,errors3_avg,running_sum3_avg=f_p_train
weights4_avg,accuracy4_avg,errors4_avg,running_sum4_avg=f_test
weights5_avg,accuracy5_avg,errors5_avg,running_sum5_avg=f_p_test


# In[16]:


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(iterations_, accuracy2_avg)
axs[0, 0].set_title('Perceptron_avg_train')
axs[0, 1].plot(iterations_, accuracy3_avg, 'tab:orange',marker='^')
axs[0, 1].set_title('PA_avg_Train')
axs[1, 0].plot(iterations_, accuracy4_avg, 'tab:green',marker='v')
axs[1, 0].set_title('Perceptron avg test')
axs[1, 1].plot(iterations_, accuracy5_avg, 'tab:red',marker='o')
axs[1, 1].set_title('PA_avg_TEST')

for ax in axs.flat:
    ax.set(xlabel='ITERATIONS', ylabel='ACCURACY')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


# #Looks similar to the ordinary perceptron. Pretty sure I am missing something.

# In[49]:


#In general this is how the accuracy curves look.
plt.plot(iterations_, accuracy2_avg,color='blue', linestyle='solid',marker='o')

plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.title("This in general is how the online accuracy curves look")
plt.show()


# ## 5.1D

# In[18]:


#split the data into sets of 100...... For this number, I am not sure I got the question right but instead of the 100 increments
# I did bigger splits of the data because with the 100 increments I could not see any real changes in the curves.
features_100=features[:100]
features_200=features[:2000]
features_300=features[:30000]
features_400=features[:50000]
feature_lists=[features_100,features_200,features_300,features_400]
labels_100=labels[:100]
labels_200=labels[:2000]
labels_300=labels[:30000]
labels_400=labels[:50000]
labels_list=[labels_100,labels_200,labels_300,labels_400]


# In[19]:


#Calling the classes with different amounts of training data, Didn't see a noticeable difference
v=Binary(1,20)
r_100=v.train(features_100,labels_100)
r_200=v.train(features_200,labels_200)
r_300=v.train(features_300,labels_300)
r_400=v.train(features_400,labels_400)
#################################################################
weights_100,accuracy_100,errors_100,running_sum_100=r_100
weights_200,accuracy_200,errors_200,running_sum_200=r_200
weights_300,accuracy_300,errors_300,running_sum_300=r_300
weights_400,accuracy_400,errors_400,running_sum_400=r_400


# In[20]:


#plots.....
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(iterations_, errors_100)
axs[0, 0].set_title('errors_100')
axs[0, 1].plot(iterations_, errors_200, 'tab:orange',marker='^')
axs[0, 1].set_title('errors_2000')
axs[1, 0].plot(iterations_, errors_200, 'tab:green',marker='v')
axs[1, 0].set_title('errors_30000')
axs[1, 1].plot(iterations_, errors_200, 'tab:red',marker='o')
axs[1, 1].set_title('errors_50000')

for ax in axs.flat:
    ax.set(xlabel='ITERATIONS', ylabel='ERRORS')


# #The curves are generally very similar. All trending downwards, However, the higher the number of examples we have, generally
# #speaking, the sooner the curve tails off. You can see the kink for the 50000 examples tails off quicker compared to the other 3

# # Multiclass Classifier

# In[21]:


#I'm pretty sure there is an easy way to create these labels for the one vs all label configuration but my loop was not 
#working right so I went the long way...............
labels0=data['label']
labels0=np.array(labels0)
labels0=np.where(labels0==0,1,-1)
labels0
labels1=data['label']
labels1=np.array(labels1)
labels1=np.where(labels1==1,1,-1)
labels1
labels2=data['label']
labels2=np.array(labels2)
labels2=np.where(labels2==2,1,-1)
labels2
labels3=data['label']
labels3=np.array(labels3)
labels3=np.where(labels3==3,1,-1)
labels3
labels4=data['label']
labels4=np.array(labels4)
labels4=np.where(labels4==4,1,-1)
labels4
labels5=data['label']
labels5=np.array(labels5)
labels5=np.where(labels5==5,1,-1)
labels5
labels6=data['label']
labels6=np.array(labels6)
labels6=np.where(labels6==6,1,-1)
labels6
labels7=data['label']
labels7=np.array(labels7)
labels7=np.where(labels7==7,1,-1)
labels7
labels8=data['label']
labels8=np.array(labels8)
labels8=np.where(labels8==8,1,-1)
labels8
labels9=data['label']
labels9=np.array(labels9)
labels9=np.where(labels9==9,1,-1)
labels9


# In[22]:


#add all lists of labels to a list of lists
list_of_labels=[labels,labels1,labels2,labels3,labels4,labels5,labels6,labels7,labels8,labels9]


# In[23]:


#Below I train 10 different classifiers based on which among the 10 classes is marked positive... Use a for loop
# So ths list of output is a list of lists,the 10 nested lists, the nested lists each have weights, errors, accuracy and 
#the running sum
list_of_outputs=[]
for i in list_of_labels:
    t=Binary()
    features=np.array(features)
    r=t.train(features,i)
    weights,accuracy,errors,running_sum=r
    list_of_outputs.append(r)


# #The code above returns ten classifiers, however after this I am not sure exactly how to proceed. All tricks I tried ended up in
# #errors...
