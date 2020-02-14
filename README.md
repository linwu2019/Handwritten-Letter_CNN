# project-01-dlwxmon-go
project-01-dlwxmon-go created by GitHub Classroom

## 1.Packages Requirements
### *We use several packages:<br/> pickle, numpy,  PIL, skimage, torch, sklearn, math, matplotlib*


    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    from PIL import Image
    from skimage.color import rgb2gray
    import skimage.filters as filt
    
    import skimage
    from skimage import filters
    
    from skimage.morphology import skeletonize, thin
    import torch.optim as optim
    import torch.nn.functional as F
   
    from sklearn.model_selection import train_test_split
    
    import torch 
    import torch.nn as nn
    from torch.autograd import Variable
    import torch.utils.data as Data
    from torch.utils.data import Dataset
    from torch.optim.lr_scheduler import MultiStepLR
    
    from sklearn.ensemble import IsolationForest

    from sklearn.svm import OneClassSVM

## 3. Files Included

 0: **Please put all files in same file or you have to add path.**
### 1 **checkpoing.pt** : the parameters are saved in this file 

  using code below to load file
  
      model.load_state_dict(torch.load('checkpoint.pt')) 
  
    
  
  
### 2 **iso_train.pickle** : this file stores the model for IsolationForests to classify unknow class as -1 laebl.

    with open('iso_train.pickle', 'rb') as f:
      clf = pickle.load(f)

### 3 **svm_train.pickle**

    with open('svm_train.pickle', 'wb') as f:
      clf1= pickle.load(clf1,f)
      
### 4. train.py

### 5. test.py




## 2. Input Format

### 1 Input format the same as train_data from last assignment.<br/>
### 2 ATTENTION： If you want to have a prediction on specific image,let us say,
    train_data[1]
 please input as format:
   
    train_data[1:2] 
Because we defalutly treat input data as a collection of images.!!
### 3 Load data：
    train_data = load_pkl('train_data.pkl')
    **PLEASE USING　VARIABLE test_data**
    test_data = load_pkl('PLEASE FILL FILE') 

## 3.Parameters
#### 1. modification split size

    train_data_, validation_data_, train_labels_, validation_labels_ = train_test_split(train_data,train_labels, test_size=0.2,shuffle=True)
   
 ### 2. SGD with momentum=0.5
 
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)
 
 ### 3. Lr,epochs,loss function
    #learningRate = 0.02 with step dicreasing by 0.7 at epochs [[6,15,20,30]]：
    scheduler = MultiStepLR(optimizer, milestones=[6,15,20,30], gamma=0.7)
    
    epoches = 50  #fixed
    batch_size = 80 # fixed
    
    #Weighed Cross Entropy Loss Function with weights =[1.4,1.4,0.8,0.8,1,0.9,0.9,1]  
    lossCriterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.4,1.4,0.8,0.8,1,0.9,0.9,1])) 
    
    # train data to get parameters
    train_validate(0.02,patience=4)
    
### 4. CNN structure
    (channels,size)
    Input :
    (1,32,32)
    
    after conv1:
    (16,32,32)
    
    after Max Pooling :
    (16,16,16)
    
    after conv2:
    (32,16,16)
    
    after Max Pooling:
    (32,8,8)
    
    full_connected layer1:
    in:32*8*8,
    out:64
    
    full_connected layer_2:
    in:64
    out:8
    
    And there are BN layers before every Relu Activation Function.
 ### 5. Isolation Forests
 
        clf=IsolationForest(n_estimators=140,behaviour='new',max_samples='auto',contamination=0.001,max_features=5)
        
 ### 6. One Class SVM
 
        clf1 = OneClassSVM(kernel='rbf',tol=0.01,nu=0.001，gamma='auto')


