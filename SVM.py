#!/usr/bin/env python
# coding: utf-8

#  # SVM

# In[1]:


#import modules
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from numpy import log2 as log
from numpy import linalg


# In[2]:


#class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,probability=False, 
#tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, 
#random_state=None)


# In[3]:


#import the data
train_df=pd.read_csv("fashion-mnist_train.csv")
test_df=pd.read_csv("fashion-mnist_test.csv")
print(train_df.shape, train_df.head(),test_df.shape,test_df.head())


# In[4]:


test_df.shape


# In[5]:


#split the traing data for training and validation,We use the first (0.8*60000) to train and the rest for validation
train=train_df.head(48000) #the first 48,000 for training
validation=train_df.tail(12000) # the last 12,000 for validation
print(train.shape, validation.shape) #checks out.
#or you can use the train, test, split function in SKlearn


# In[16]:


#split into features and labels.
train_features=preprocessing.scale(np.array(train.drop("label",axis=1)))
validation_features=preprocessing.scale(np.array(validation.drop("label",axis=1)))
train_labels=np.array(train["label"])
validation_labels=np.array(validation["label"])
test_features=preprocessing.scale(test_df.drop("label",axis=1))
test_labels=test_df["label"]


# In[30]:


#create a list for the regularization parameter "C", we'll call that list regularization.
reg=[10**-4,10**-3,10**-2,10**-1,10**0]
# reg2=[10**0,10**1,10**2,10**3,10**4]


# In[24]:


# clf_list1=[]
# for i in reg:
#     clf = svm.SVC(C=i,kernel='linear',max_iter=100000) #linear kernel
#     clf_list1.append(clf)
# print(clf_list1)


# In[25]:


# class_list1=[]
# for j in clf_list1:
#      fit_classifier=j.fit(train_features,train_labels)
#      class_list1.append(fit_classifier)


# #### predict on each of the 10 classifiers and predict accuracy.

# In[ ]:


# for classifier in class_list1:
#     predict_train=classfier.predict(train_features)
#     predict_validation=classifier.predict(validation_features)
#     predict_test=classifier.predict(test_features)


# In[9]:


# #create a classifier
# clf_list=[]
# for i in reg:
#     clf = svm.SVC(C=i,kernel='linear') #linear kernel
#     clf_list.append(clf)
# print(clf_list)


# In[10]:


# #train on each of the classifiers you have built
# for j in clf_list:
#     j.fit(train_features,train_labels)


# In[22]:


#use 30% of the data and then scale it to see whether is trains better
train30=train_df.head(15000)
validation30=train_df.tail(3000)
print(train30.shape, validation30.shape)


# In[23]:


#create training, validation and testing features.
train30_features=preprocessing.scale(train30.drop("label",axis=1))
validation30_features=preprocessing.scale(validation30.drop("label",axis=1))
train30_labels=train30["label"]
validation30_labels=validation30["label"]
test_features=preprocessing.scale(test_df.drop("label",axis=1))
test_labels=test_df["label"]


# In[31]:


#create a classifier
clf_list=[]
for i in reg:
    clf = svm.SVC(C=i,kernel='linear') #linear kernel
    clf_list.append(clf)
print(clf_list)


# In[32]:


class_list=[]
for j in clf_list:
     fit_classifier=j.fit(train30_features,train30_labels)
     class_list.append(fit_classifier)


# In[33]:


print(class_list)


# #### Predict with all the classifiers we have created

# In[36]:


train_prediction_list=[] 
test_prediction_list=[]
validation_prediction_list=[]
for i in class_list:
        predict_train=i.predict(train30_features)
        train_prediction_list.append(predict_train) 


# #### Find accuracy for all the classifiers on the training set

# In[40]:


Train_accu=[]
for j in train_prediction_list:
    a=metrics.accuracy_score(train30_labels,j )
    Train_accu.append(a)
    
print(Train_accu)


# #### For testing set.

# In[41]:


for i in class_list:
        predict_test=i.predict(test_features)
        test_prediction_list.append(predict_test)         


# In[42]:


testaccu=[]
for j in test_prediction_list:
    b=metrics.accuracy_score(test_labels,j )
    testaccu.append(b)
    
print(testaccu)


# #### For validation set

# In[43]:


for i in class_list:
    predict_validation=i.predict(validation_features)
    validation_prediction_list.append(predict_validation)


# In[46]:


vaccu=[]
for j in validation_prediction_list:
    c=metrics.accuracy_score(validation_labels,j )
    vaccu.append(c)
    
print(vaccu)


# ##### Plot accuracies as a function of the C's

# In[76]:


plt.plot(reg,Train_accu,marker='o')
plt.ylabel('Training Accuracy')
plt.xlabel('Regularization Parameter')
plt.title("Training Accuracy vs.Regularization Parameter ")
plt.show()


# In[78]:


plt.plot(reg,testaccu,'tab:red',marker='o')
plt.ylabel('Testing Accuracy')
plt.xlabel('Regularization Parameter')
plt.title("Testing Accuracy vs.Regularization Parameter ")
plt.show()


# In[79]:


plt.plot(reg,vaccu,'tab:purple',marker='o')
plt.ylabel('Validation Accuracy')
plt.xlabel('Regularization Parameter')
plt.title("Validation Accuracy vs.Regularization Parameter ")
plt.show()


# #### According to the validation set, a c of 0.01 is the best, so we train the SVM on this value of C

# In[61]:


# train30_pred=class_list[0].predict(train30_features)
# validation30_pred=class_list[0].predict(validation30_features)
test_pred=class_list[0].predict(test_features)


# In[48]:


# print("Training Accuracy, C=0.0001:",metrics.accuracy_score(train30_labels,train30_pred ))
# print("Validation Accuracy, C=0.0001:",metrics.accuracy_score(validation30_labels,validation30_pred ))
# print("Test Accuracy, C=0.0001:",metrics.accuracy_score(test_labels,test_pred ))


# In[49]:


#for the training features:
# train_predlist=[]
# for i in class_list:
#     predict=i.predict(train30_features)
#     train_predlist.append(predict)


# In[50]:


#accuracy for the training set:
#print(train_predlist)
# accuracy_train=[]
# for i in train_predlist:
#     accuracy=metrics.accuracy_score(train30_labels,i)
#     accuracy_train.append(accuracy)
# print(accuracy_train)


# In[51]:


#for the validation features:
# V_predlist=[]
# for i in class_list:
#     predict_V=i.predict(validation30_features)
#     V_predlist.append(predict_V)


# In[52]:


#accuracy for the validation set:
# validation_accuracy=[]
# for i in V_predlist:
#     accuracy_V=metrics.accuracy_score(validation30_labels,i)
#     validation_accuracy.append(accuracy_V)
# print(validation_accuracy)


# In[32]:


#for the testing features
# test_predlist=[]
# for i in class_list:
#     predict_test=i.predict(test_features)
#     test_predlist.append(predict_test)


# In[53]:


#accuracy for the testing set....
# testing_accuracy=[]
# for i in test_predlist:
#     accuracy_test=metrics.accuracy_score(test_labels,i)
#     testing_accuracy.append(accuracy_test)
# print(testing_accuracy)


# #### Get support vectors. This is an example of how you get support vectors for one of the classifiers that you train.

# In[67]:


d=class_list[0].support_vectors_


# #### And then this is how you plot the support vectors

# In[68]:


plt.plot(d)
plt.show()


# In[54]:


#the best value for c according to the validation set is C=0.01, also test set indicates same
trainingsetC=train_df.head(15000)
trainingsetC.append(train_df.tail(3000))
trainingsetC_features=preprocessing.scale(trainingsetC.drop("label",axis=1))
trainingsetC_labels=trainingsetC["label"]
#need labels and features
clf_C=svm.SVC(C=0.01,kernel='linear')


# In[55]:


clf_C.fit(trainingsetC_features,trainingsetC_labels)


# In[56]:


#testing accuracy on classifier fir in the best value of C
test_predict_c=clf_C.predict(test_features)


# #### 2b.Testing accuracy and confusion matrix

# In[57]:


#print the accuracy score based on the testing set based on the best value of C
print("Accuracy score on best value of C:",metrics.accuracy_score(test_labels,test_predict_c))


# In[58]:


class_names=test_labels.unique()
print(class_names)


# ### Confusion Matrix

# In[59]:


#confusion matrix for SVM classifier
title='confusion matrix for SVM classifier'
disp = plot_confusion_matrix(clf_C, test_features, test_labels,display_labels=class_names,cmap=plt.cm.Blues)
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()


# ### part c

# In[52]:


#SVM's with the RBF Kernel of degree 2, 3 and 4
clf_2 = svm.SVC(C=0.01,kernel='rbf',degree=2)
clf_3 = svm.SVC(C=0.01,kernel='rbf',degree=3)
clf_4 = svm.SVC(C=0.01,kernel='rbf',degree=4)


# In[53]:


#fit all on the trainng set for the degree 2 polynomial.
clf_2.fit(trainingsetC_features,trainingsetC_labels)
clf_3.fit(trainingsetC_features,trainingsetC_labels)
clf_4.fit(trainingsetC_features,trainingsetC_labels)


# In[54]:


#predict on the training set and find accuracies
training2=clf_2.predict(trainingsetC_features)
training3=clf_3.predict(trainingsetC_features)
training4=clf_4.predict(trainingsetC_features)
print("Accuracy degree 2,3 and 4: Training set:")
print(metrics.accuracy_score(trainingsetC_labels,training2))
print(metrics.accuracy_score(trainingsetC_labels,training3))
print(metrics.accuracy_score(trainingsetC_labels,training4))


# In[56]:


#predict on the testing set and find accuracies
test2=clf_2.predict(test_features)
test3=clf_3.predict(test_features)
test4=clf_4.predict(test_features)
print("Accuracy degree 2,3 and 4:Testing set:")
print("2",metrics.accuracy_score(test_labels,test2))
print("3",metrics.accuracy_score(test_labels,test3))
print("4",metrics.accuracy_score(test_labels,test4))


# In[57]:


#predict on the validation set and find accuracies
validation2=clf_2.predict(validation30_features)
validation3=clf_3.predict(validation30_features)
validation4=clf_4.predict(validation30_features)
print("Accuracy degree 2,3 and 4:Validation set:")
print(metrics.accuracy_score(validation30_labels,validation2))
print(metrics.accuracy_score(validation30_labels,validation3))
print(metrics.accuracy_score(validation30_labels,validation4))


# #### For all the training, test and validation examples, there does not seem to be any change when we change the degree of the polynomial?? 

# # 2. Kernelized perceptron

# #### For this part we make a small change to the multiclass classifier we built in HW 1. So It's prudent to say that this code is adopted from Taha's code

# In[157]:


def linear_kernel(x1,x2):
    np.dot(x1,x2)
    
#polynomial Kernel(x,y,p=3)
def polynomial_kernel(x,y,p=2):
    return (1+np.dot(x, y)) ** p

#RBF Kernel
def rbf_kernel(x,y,sigma=0.5):
    return np.exp(-lnalg.norm(x-y)**2/(2*(sigma**2)))


# #### Accroding to the last exercise plynomial of degree 2 is the best and so we use it for modifying the function below

# In[158]:


def Fk(kt, x, k):
    """
    F function for multi_class online classifier.
    :param kt: Total number of classes.
    :param x: Data input.
    :param k: Class label.
    :return F(x, k).
    """
    
    F = np.zeros((kt, x.shape[0]))
    F[k, :] = x
    F=(1+np.dot(F, F.T))**2 # this is the kernelization part.
     
    return F


# In[159]:


def multi_class_classifier(X, y, kt, T, PA=False):
    """
    Implementation of the Online Multi-class CLassifier in Algorithm 2.
    :param D: Training examples.
    :param kt: Number of classes.
    :param T: Maximum number of iterations.
    :param PA: Boolean for use or not of PA approach.
    :return: Weight vector generated.
    """
    w = 0
    for _ in range(T):
        D = zip(X, y)
        for xt, yt in D:
            y_hat = np.argmax(np.sum(w * Fk(kt, xt, yt), axis=1))
            if y_hat != yt:
                if PA==True:
                    tau = (1 - (np.sum(w * Fk(kt, xt, yt), 
                           axis=1) - np.sum(w * Fk(kt, xt, y_hat), 
                           axis=1)))/(np.linalg.norm(Fk(kt, 
                                   xt, yt) - Fk(kt, xt, y_hat),ord=2)**2)
                    tau = np.expand_dims(tau, axis=1)
                else:
                    tau = 1                    
                w += tau * (Fk(kt, xt, yt) - Fk(kt, xt, y_hat))
    return w


# ##### Train for five iterations

# In[160]:


vector=multi_class_classifier(train30_features, train30_labels, 10, 5)


# In[161]:


D=zip(train_features,train_labels)


# In[162]:


kt=10
def accuracy_fn(D, w, ll):
    """
    Computes the accuracy of a given model's weight vector.
    :param D: Testing examples.
    :param w: Weight vector.
    :param ll: Total length of the testing data.
    :return: Accuracy between 0 and 1.
    """
    score = 0
    for xt, yt in D:
        if yt == np.argmax(np.sum(w * Fk(kt, xt, yt), axis=1)):
            score += 1
    return score/ll


# ##### accuracy on the training features is 89%

# In[163]:


accuracy_fn(D, vector,train_features.shape[0] )


# #### Accuracy on the validation Set is 90%

# In[164]:


D2=zip(validation_features,validation_labels)


# In[165]:


accuracy_fn(D2, vector,validation_features.shape[0] )


# #### Accuracy on the testing set is

# In[166]:


D3=zip(test_features,test_labels)


# In[167]:


accuracy_fn(D3, vector,test_features.shape[0] )


# ## 3. Breast Cancer classification using Decision trees.

# In[2]:


#read in the cancer data
cancer_data=pd.read_csv("wdbc.data")
cancer_data.shape #The data has 568 examples and 32 attributes.
cancer_data.head() # Let's rename these columns:


# In[3]:


#rename the column
cancer_data=cancer_data.rename(columns={"M": "target", "17.99": "rad_mean","10.38":"tex_mean","122.8":"per_mean","1001":" area_mean","0.1184":"smo_mean","0.2776":"comp_mean","0.3001":"conc_mean","0.1471":"cp_mean","0.2419":"sy_mean","0.07871":"fd_mean","1.095":"rad_sd","0.9053":"text_sd","8.589":"per_sd","153.4":"area_sd","0.006399":"smo_sd","0.04904:":"com_sd","0.05373":"conc_sd","0.01587":"Concp_sd","0.03003":"fd_sd","0.006193":"rad_worst","25.38":"tex_worst","17.33":"per_worst","184.6":"area_worst","2019":"smo_worst","0.1622":"com_worst","0.6656":"conc_worst","0.7119":"cp_worst","0.2654":"sym_worst","0.4601":"fcd_worst"})
# cancer_data.rename(columns={"0.1184":"smo_mean","0.2776":"comp_mean","0.3001":"conc_mean","0.1471":"cp_mean","0.2419":"sy_mean","0.07871":"fd_mean"})


# In[4]:


#drop the ID column
cancer_data=cancer_data.drop(['842302'], axis=1)# drop the ID column


# ###### Sklearn's webpage has  a good description of the data.See here: https://scikitlearn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

# In[5]:


#70% for training, the next 10% for validation and the last 20 for testing
train_set=cancer_data.head(398)
test_set=cancer_data.tail(114)
validation_set=cancer_data[399:454]
print(train_set.shape,test_set.shape,validation_set.shape)#70,last 20, middle 10%
train_set.head()


# In[6]:


#‘eps’ here is the smallest representable number
eps = np.finfo(float).eps


# ### Making a decision tree
# #### 1.Compute entropy for data set, calculate entropy for all categorical values, take avergae info entropy for current attribute
# #### 2.Calculate gain for current attribute
# #### 3.Pick the highest gain attribute
# #### 4.iterate until desired tree

# In[7]:


#calculate entropy for the whole data set
entropy_0=0
classes=train_set["target"].unique()
for value in classes:
    proportion=train_set["target"].value_counts()[value]/len(train_set["target"])
    entropy_0+=-proportion*np.log2(proportion)
print(entropy_0) #so the entropy at the root node is ~99%


# In[8]:


# we define a function to calculate the entropy for each attribute in the data set
def entropy(dataframe, attribute):
    labels= train_set["target"].unique() # M and B as shown above
    variables=train_set[attribute].unique() # gives different numbers in the attribute or feature
    
    attribute_entropy=0
    
    for variable in variables:
        entropy_each_feature=0
        for label in labels:
            num = len(train_set[attribute][train_set[attribute]==variable][train_set["target"] ==label])#numerator
            den = len(train_set[attribute][train_set[attribute]==variable])
            proportion= num/(den+eps)
            entropy_each_feature += -proportion*log(proportion+eps) #calculates entropy for each feature
        proportion_2=den/len(train_set)
        attribute_entropy+=-proportion_2*entropy_each_feature
    return(abs(attribute_entropy))


# In[9]:


#These are the entropies for all the features in the data set.
a_entropy = {k:entropy(train_set,k) for k in train_set.columns[1:]}
a_entropy


# #### 3a. Calculate the information gain for every feature.

# In[10]:


#Calculate the information gain for each attribute. INFO_GAIN=entropy of dataset - entropy of attribute
def info_gain(ent_dataset, ent_attr):
    return (ent_dataset-ent_attr)
Info_Gain={k:info_gain(entropy_0,a_entropy[k]) for k in a_entropy}  
print(Info_Gain)


# In[11]:


#this function takes in the dataframe and tells us which is the most important feature to split on.
def most_imp(df):
    Entropy_att = []
    IG = []
    for column in df.columns[1:]:
       
        IG.append(entropy_0-entropy(df,column))
    return df.columns[1:][np.argmax(IG)]


# In[12]:


#feed in the data set and it tells us which is the most important feature to split on
most_imp(train_set)


# In[14]:


#after the split we get the resulting tree.
def sub_tree(df,node,value):
    return df[df[node]==value].reset_index(drop=True)


# #### Build the tree: With the ID3 algorithm, we make a loop to recursively call the function at every node and only stop if the node is pure.

# In[13]:


def Tree(df,tree=None): # defaulted to none
    Class=df['target']
    #bulding decsion tree, using the "most_imp" function above, get the feature with that has the most important split.
    node=most_imp(df) # get feature with the maximum information gain
    attValue = np.unique(df[node]) # get distinct values of that node
    #make a dictionary to create the tree:
    if tree is None:
        current_node=tree[node]
    for value in attValue:
        subtree=sub_tree(df,node,value)
        clValue,counts = np.unique(subtree['target'],return_counts=True)
        
        if len(counts)==1: #checking the purity of the node
            tree[node][value]=clValue
        else:
            tree[node][value]==Tree(subtree) # recursive call
    return tree
        

