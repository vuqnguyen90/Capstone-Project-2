# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:24:23 2020

@author: Vu.Nguyen
"""

# Import necessary libraries 
import numpy as np 
from keras.models import Sequential 
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.layers import LeakyReLU
import os 
import FunctionLists as f 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from keras.callbacks import Callback,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.utils import resample
import pandas as pd 
import statsmodels.api as sm 

import math as m
import seaborn as sns 

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import classification_report,confusion_matrix

from keras.regularizers import l1

from collections import Counter


###################################### Function to preprocess data. ###########################################
# Prepare the Dataframes for Features and Targets
def preprocess_data(ss,unwanted): 
    fn=os.listdir(ss)
    data=f.load_ss_csv_to_df(fn,ss)

    data[['Depth','SSNS_Prefrac','SSNS_Prefrac_RMS','SSNS_Duringfrac','SSNS_Duringfrac_RMS','SSNS_Afterfrac_RMS','SSNS_Afterfrac','SSPS_Prefrac','SSPS_Prefrac_RMS','SSPS_Duringfrac_RMS','SSPS_Afterfrac_RMS','SSPS_Duringfrac','SSPS_Afterfrac','Total_Strain_Duringfrac','Total_Strain_Afterfrac']]=data[['Depth','SSNS_Prefrac','SSNS_Prefrac_RMS','SSNS_Duringfrac','SSNS_Duringfrac_RMS','SSNS_Afterfrac_RMS','SSNS_Afterfrac','SSPS_Prefrac','SSPS_Prefrac_RMS','SSPS_Duringfrac_RMS','SSPS_Afterfrac_RMS','SSPS_Duringfrac','SSPS_Afterfrac','Total_Strain_Duringfrac','Total_Strain_Afterfrac']].astype(float)

    data['FracHits']=data['FracHits'].astype(int)
    data.set_index('Depth',inplace=True)

    # Create Train and Test Data
    target=data['FracHits']

    features=data.drop(unwanted,axis=1)
    
    return data,features,target

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

##################################### Main File ################################################################
#
#
ss=r'C:\Users\vu.nguyen.SILIXA\Desktop\MRO PrimeRue\Capstone Project\PXD_NES_NEN\SlowStrain_v4\NES114H\With_FracHits_Assigned\1M\Train\\'
unwanted=[ 'SSNS_Prefrac', 'SSNS_Prefrac_RMS', 'SSNS_Duringfrac',
       'SSNS_Duringfrac_RMS',
       'SSPS_Prefrac', 'SSPS_Prefrac_RMS', 
       'SSPS_Afterfrac_RMS', 'SSPS_Afterfrac',
        ]
data,features,target=preprocess_data(ss,unwanted)


# To Separate normal from anomalous data
no_hit=features.loc[features['FracHits']==0]
hits=features.loc[features['FracHits']==1]

y_no_hit=no_hit['FracHits']
no_hit=no_hit.drop(['FracHits'],axis=1)

y_hits=hits['FracHits']
hits=hits.drop(['FracHits'],axis=1)

# To split "normal data" into the training and validation dataset 
# Leave anamalous data as holdout dataset
x_train,x_val=train_test_split(no_hit,test_size=0.1, random_state=10)
x_unseen=np.array(hits)

# To scale the data
scaler=StandardScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_val_scaled=scaler.transform(x_val)
x_unseen_scaled=scaler.transform(x_unseen)

################################# EDA ######################################################################

data.head()

data.shape
data.info()
data.describe()

# Draw heatmap correlation 
data_corr=data.corr(method='pearson')
plt.figure()
sns.heatmap(data_corr, annot=True)
plt.xticks(rotation=45)


# Draw boxplot for all columns in one chart
data.plot(kind='box')
plt.xticks(rotation=45)


# Draw KDE 
plt.figure()

for i in range(data.shape[1]):
    plt.figure()
    sns.distplot(data.iloc[:,i],hist=True, kde=True,bins=100)
    
    
sns.pairplot(data)  

################################# To Build Autoencoder model ############################################### 
s=(x_train_scaled.shape)[1]
input_fh=Input(shape=(s,))

encoded = Dense(units=12,activity_regularizer=l1(0.2))(input_fh)
encoded=LeakyReLU(alpha=0.3)(encoded)

decoded = Dense(units=s)(encoded)
decoded = LeakyReLU(alpha=0.3)(decoded)

autoencoder=Model(input_fh, decoded)

encoder=Model(input_fh,encoded)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[get_f1])


autoencoder.fit(x_train_scaled,x_train_scaled,
                epochs=50,
                batch_size=2,
                shuffle=True,
                validation_data=(x_val_scaled, x_val_scaled))


################################# To show how well the model was trained ############################################### 
plt.figure()
plt.plot(autoencoder.history.history['loss'])
plt.plot(autoencoder.history.history['val_loss'])
plt.title("Model's Training & Validation loss across epochs")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.figure()
plt.plot(autoencoder.history.history['acc'])
plt.plot(autoencoder.history.history['val_acc'])
plt.title("Model's Training & Validation loss across epochs")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


################################ To compare original data set with reconstruction data ########################### 

# To calculate the total error squared for training set
N = len(x_train_scaled)
max_se = 0.0; max_ix = 0
curr_se=[]; idx=[]
train_pred=autoencoder.predict(x_train_scaled)
for i in range(N):
  diff=( x_train_scaled[i] - train_pred[i])
  curr_se.append(np.sum(diff * diff))
  idx.append(i)

df_temp=x_train.copy()
df_temp.reset_index(inplace=True)
mse_dept=df_temp.loc[idx,'Depth'].tolist()


# To calculate the total error squared for validation set
N = len(x_unseen_scaled)
max_se = 0.0; max_ix = 0
curr_se2=[];idx2=[]
unseen_pred=autoencoder.predict(x_unseen_scaled)
for i in range(N):
  diff = x_unseen_scaled[i] - train_pred[i]
  curr_se2.append(np.sum(diff * diff))
  idx2.append(i)

y_hits_df=y_hits.to_frame()
y_hits_df.reset_index(inplace=True)
df_temp2=y_hits_df.copy()
df_temp2.reset_index(inplace=True)
mse_dept2=df_temp2.loc[idx2,'Depth'].tolist()



# To compare the original input of training dataset to reconstructed input
x_train_scaled_df=pd.DataFrame(x_train_scaled)
x_train_scaled_df.plot(subplots=True)
train_pred_df=pd.DataFrame(train_pred)
train_pred_df.plot(subplots=True)

# To compare the original input of validation dataset to reconstructed input
x_unseen_scaled_df=pd.DataFrame(x_unseen_scaled)
x_unseen_scaled_df.plot(subplots=True)
unseen_pred_df=pd.DataFrame(unseen_pred)
unseen_pred_df.plot(subplots=True)


############################ Test the Model with Holdout dataset ##############################

#Load Holdout dataset 
ss=r'C:\Users\vu.nguyen.SILIXA\Desktop\MRO PrimeRue\Capstone Project\PXD_NES_NEN\SlowStrain_v4\NES114H\With_FracHits_Assigned\1M\Unseen\\'
data_unseen,features_unseen,target_unseen=preprocess_data(ss,unwanted)

features_unseen=features_unseen.drop('FracHits',axis=1)
scaler=StandardScaler()
scaler.fit(features_unseen)

features_unseen_scaled=scaler.transform(features_unseen)


# Calculate the total squared error between the reconsructed input vs input for holdout data
N = len(features_unseen_scaled)
max_se = 0.0; max_ix = 0
curr_se3=[];idx3=[]
unseen_pred2=autoencoder.predict(features_unseen_scaled)

curr_se3=[]
for i in range(N):
  diff = features_unseen_scaled[i] - unseen_pred2[i]
  curr_se3.append(np.sum(diff * diff))
  idx3.append(i)

fh=[]
fh_idx=[]

for i,mse in enumerate(curr_se3):
        fh.append(mse)
        fh_idx.append(i)


########################################## To Plot the squared error for normal data vs anomalous data ######
# with predefined threshold 
        
plt.figure()
plt.scatter(idx,curr_se,label='Non-Frac-Hits Data')
plt.scatter(idx2,curr_se2,label='Frac-Hits Data')
plt.ylabel('MSE')
plt.xlabel('Data Point')
plt.legend()
   
plt.scatter(fh_idx,fh,label='Unseen Test Data') 
plt.plot([0,6200],[1.9,1.9],color='r',label='Threshold')
plt.legend()




