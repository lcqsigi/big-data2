#!/usr/bin/env python
# coding: utf-8

# # Regression Example in Keras
# Predicting house prices in Boston, Massachusetts 

# #### Load dependencies

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
#import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D 
from tensorflow.keras.layers import BatchNormalization 
#from sklearn.model_selection import train_test_split


# #### Load data

# In[2]:


df=pd.read_csv('amazon.csv')
#(X_train, y_train), (X_valid, y_valid) = boston_housing.load_data()


# In[3]:


shuffle=np.random.permutation(len(df))
test_size=int(len(df)*0.2)
test_aux=shuffle[:test_size]
train_aux=shuffle[test_size:]
train_df=df.iloc[train_aux]
test_df=df.iloc[test_aux]


# In[4]:


#train_df.shape


# In[5]:


#test_df.shape


# In[6]:


X_train=train_df.to_numpy()
X_train_2=X_train[:,1:]
print(X_train_2.shape)


# In[7]:


X_test=test_df.to_numpy()
X_test_2=X_test[:,1:]


# In[8]:


y_train=train_df.to_numpy()


# In[9]:


list1=[]
for x in range(0,25600,1):
    list1.append(y_train[x][0])
print(list1)


# In[10]:


y_vector=np.asarray(list1)
#y_train_2=np.transpose(y_vector)
y_train_2=np.array([y_vector]).T
print(y_train_2.shape)


# In[11]:


y_test=test_df.to_numpy()


# In[12]:


list2=[]
for x in range(0,6400,1):
    list2.append(y_test[x][0])
print(list2)


# In[13]:


y_vector=np.asarray(list2)
#y_train_2=np.transpose(y_vector)
y_test_2=np.array([y_vector]).T
print(y_test_2.shape)


from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Reshape,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Conv1D,MaxPooling1D
import numpy as np
import os

#(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

def build0():
    model=Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(36,)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build1():
    model=Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(36,)))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

def build2():
    model=Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(36,)))
    model.add(Dense(64,activation='selu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='selu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

def build3():
    model=Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(36,)))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

def build4():
    model=Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(36,)))
    model.add(Reshape((36,1)))
    model.add(Conv1D(filters=13,strides=1,padding='same',kernel_size=2,activation='relu'))
    model.add(Conv1D(filters=26, strides=1, padding='same', kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
    model.add(Conv1D(filters=52, strides=1, padding='same', kernel_size=2, activation='relu'))
    model.add(Conv1D(filters=104, strides=1, padding='same', kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

def build7():
    model = Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(64,)))
    model.add(Reshape((64, 1,1)))
    model.add(Conv2D(filters=13, strides=1, padding='same', kernel_size=1, activation='relu'))
    model.add(Conv2D(filters=26, strides=2, padding='same', kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
    model.add(Conv2D(filters=52, strides=1, padding='same', kernel_size=1, activation='relu'))
    model.add(Conv2D(filters=104, strides=2, padding='same', kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

def build6():
    model = Sequential(name='boston')
    model.add(Dense(64,input_shape=(36,),activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1,activation='linear'))
    return model

def build5():
    model = Sequential(name='boston')
    model.add(Dense(128,input_shape=(36,),activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1,activation='linear'))
    return model

for i in range(5,7):
    model=eval("build"+str(i)+"()")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.compile(optimizer,'mae')#3.0895
    history=model.fit(X_train_2,y_train_2,batch_size=16,epochs=10000,verbose=False,validation_data=(X_test_2,y_test_2))
    print(history.history)

    list3=[]
    for x in range(0,6400,1):
        list3.append(str(y_test_2[x])[1:-1])


    list4=[]
    for x in range(0,6400,1):
        list4.append(str(model.predict(np.reshape(X_test_2[x],[1,36])))[2:-2])


    f=open("result2.txt",'a')
    f.write(str(history.history['val_loss'][-1])+"\n")
    f.write(str(list3)+"\n")
    f.write(str(list4)+"\n")
    f.close()
# In[15]:

