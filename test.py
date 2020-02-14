############################################################
import pickle
  
import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image
from skimage.color import rgb2gray
import skimage.filters as filt

# import scipy.misc
# import scipy

import skimage
from skimage import filters

from skimage.morphology import skeletonize, thin
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import sklearn


import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.ensemble import IsolationForest
##########################################################################################################
#  Define Function and class
def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)


def show_train_data(start,end,columns):
    num_show = end-start+1   #  how many pictures you are gonna show/USE TO DEBUG
#     num_show = 100
#     columns=5
    rows = math.ceil(num_show/columns)  
#     print(rows)
    fig,ax = plt.subplots(rows,columns,figsize=(12,12))
    for i in range(start,end):
        shape = train_data[i].shape
#         print(shape)
        ax= plt.subplot(rows,columns,i+1-start)
        ax.imshow(train_data[i])
#         ax[1,1].axis('off')
        ax.set_title(f'{i}')
        ax.axis('off')


def centralize_image(fg,debug=True):   # centralize data

    # load image
    # find bounds
    nz_r,nz_c = fg.nonzero() #return non zeros r= row c = colunm
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1 # left and right boundary for characters
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1 #  top and buttom

    # extract window
    win = fg[t:b,l:r]

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255) #
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win
    # print(out_win)

    #save out result as numpy array

    return out_win




def stretch(train_data):                  # stretch data to the same size : 32 by 32
    im = np.array(train_data,dtype=np.uint8)
    # im = im.astype(np.uint8)
    im = centralize_image(im) # centralized 
    img = np.array(im,dtype=np.double)
    new_img=skimage.transform.resize(img,(32,32), mode='constant',anti_aliasing=False)  
    #print(new_img)
    thresh = filt.threshold_otsu(new_img)
    bi_img = new_img > thresh
    return bi_img




# specialize to show the bi_data
def show_stretch(bi_img):
    bi_img = stretch(bi_img)
    # print(bi_img)
    plt.imshow(bi_img,cmap=plt.cm.gray)
    plt.show()





# debug : stretch then thin the image
def show_thin(image):
    # first stretch
    img=stretch(image)
  
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original', fontsize=20)

    ax[1].imshow(thin(img), cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Thinned', fontsize=20)

    fig.tight_layout()
    plt.show()



# convert image to tensor that fits into CNN
def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

def show_batch(epoches,loader):
    for epoch in range(epoches):   # train
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())
            
def preprocessing(data): # Turn the Data to Tensor type .
    a=[]
    for i in range(len(data)):
        a.append(np.asarray(thin(stretch(data[i])))) # do the centralize and strectch and thinning 
    a1=np.asarray(a)
    return torch.Tensor(a1).view(len(data),1,32,32) # add one dimension to make it suitable to put in dataloader







# Early stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...','\n')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# Initialize weights and bias
def Init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))  # if it is convolution layer  for Relu actionvation 
            if m.bias is not None:            # applying "He initializaiton"
                m.bias.data.zero_()            # and set bias to 0 
        elif isinstance(m, nn.BatchNorm2d):         # set BN layer to w=1,b=0
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):     # set full connected layer to b=0, w belongs to nomal distibution 
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):     # set BN layer to w=1,b=0
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# model 
class LetterCNN(nn.Module):
    def __init__(self):
        super(LetterCNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d( 
                in_channels=1, # only 1 channel
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
                    ), # shape 16,32,32 (channels,size)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)# shape (16,16,16)
            ) 

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
                    ),#(32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) #(32,8,8)
            )
        


        self.fc1=nn.Linear(32*8*8,64) # fc1 full-connected layer 1
        self.bn = nn.BatchNorm1d(64)  # batch normalization
        self.relu = nn.ReLU()      # relu
        self.fc2=nn.Linear(64,8)    # full connected layer 2


    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
      
        x=x.view(x.size(0),-1)    # flatten the output to (batch_size, 32 * 8 * 8)

        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
      # x = F.relu(x)
        x = self.fc2(x)       # 
        return  x  # return x     this is unormalizaed x


def train_validate(learningRate,patience,momentum=0.5):
    # train and validate epoches
    model = LetterCNN()
    # weights initialization
    # Init_weights(model)
    model.apply(Init_weights)

    # initalization trainning lists
    avg_loss_per_epoch_list=[]
    loss_temp =[]   # for batch upadating
    loss_list=[]            # record every last batch's loss for each epoch

    accuracy_per_epoch_list=[]  
    accuracy_temp=[]
    accuracy_list=[]          ## record every last batch's accuracy for each epoch

    # initialization validation lists
    val_loss_list=[]
    val_accuracy_list=[]
    val_avg_loss=[]

    iterations=len(train_loader) # how many steps/batches in one epoch


    # define the loss function and optimizer
    lossCriterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.4,1.4,0.8,0.8,1,0.9,0.9,1]))  # we use crossentropy loss criterion
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)   # momentum method /lr=0.01 before

    scheduler = MultiStepLR(optimizer, milestones=[6,15,20,30], gamma=0.7)

    # Early stopping start
    # patience = 6
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epoches):  # epoches
    # train mode
        model.train()
        for iteration,(images,labels) in enumerate(train_loader): # for each step/block for training loader

            outputs = model(images)
# collect the loss, last batch' loss and average loss for epoch
            loss = lossCriterion(outputs,labels) # get loss for every step
            loss_temp.append(loss.item())
            loss1=loss.item()



			#update weights and do BP
            optimizer.zero_grad()  # To avoid gradient sums
            loss.backward()     # back propagation
            optimizer.step()    #All optimizers implement a step() method, that updates the parameters.

			# print(len(labels))
            total = labels.size(0)  # how many labels do you have in this step(batch)
            pro8=F.softmax(outputs,dim=1).data
            _,predicted = torch.max(pro8,1) # return the prdicted indices for each row


			# print(outputs.data)
			# .sum()is used to calculate # of elements whose predicts are same as labels 
			#but it return in term of tensor, we use item() to retrieve number in it.
			# print((predicted == labels).sum())

			# collect accuracy list for train data
            correct = (predicted == labels).sum().item()

            accuracy_temp.append(correct/total)  # for bacthes

            acc = correct/total # record accuracy instantly


		# print(loss_temp)

        accuracy_list.append(acc)  # record every last batch's Accuracy of each epoch
        accuracy_per_epoch_list.append(np.average(accuracy_temp)) # record all batch's average ACCuracy of each epoch

        loss_list.append(loss1)   # record every last batch's LOSS of each epoch

        avg_loss_per_epoch_list.append(np.average(loss_temp))   #  record all batch's average LOSS of each epoch



			# if (iteration+1) % iterations ==0: # track all the statistics/10 batches per track
		# print('Trainmodel Epoch[{}/{}],AvgLoss:{:.4f},AvgAccuracy:{:.2f}%'.format(epoch+1,epochs,loss_list[epoch],accuracy_list[epoch]*100))

        print('Trainmodel Epoch[{}/{}],   Loss:{:.4f},   Accuracy:{:.2f}%'.format(epoch+1,epoches,loss1,acc*100))
		# print(len(accuracy_list))



		### validation##############################################################################################
        model.eval()
        for j,(images,labels) in enumerate(validation_loader):   # loader with all the data

            outputs = model(images)
			# print(outputs.shape)
            _,predicted = torch.max(F.softmax(outputs,dim=1),1)

            correct_val = (predicted == labels).sum().item()
            total_val = labels.size(0)

            val_accuracy_list.append(correct_val/total_val)
            val_loss = lossCriterion(outputs,labels)
            val_loss_list.append(val_loss.item())

		  # early_stopping needs the validation loss to check if it has decresed, 
			# and if it has, it will make a checkpoint of the current model 



		#  averageLoss
        val_avg_loss.append(np.average(val_loss_list))
		# clear temp lists to track next epoach
        accuray_temp=[]
        loss_temp=[]

        print('Validation Epoch[{}/{}]:,  Loss:{:.4f},   Accuracy:{:.2f}%\n'.format(epoch+1,epoches,val_avg_loss[epoch],val_accuracy_list[epoch]*100))


		# using average loss to do early stopping 
        early_stopping(np.average(val_loss_list), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break  
        val_loss_list=[]

        scheduler.step()


    # checkpoint
    model.load_state_dict(torch.load('checkpoint.pt'))


    # plot the accuracy and loss
    fig = plt.figure(num=2, figsize=(15, 8),dpi=80)
    ax1 = fig.add_subplot(2,1,1)  
    ax2 = fig.add_subplot(2,1,2)

    ax1.plot(range(len(accuracy_list)),accuracy_list,color='g',label='Train_Accuracy')
    ax1.plot(range(len(val_accuracy_list)),val_accuracy_list,color='r',label='Validation_Accuracy')

    ax2.plot(range(len(loss_list)),avg_loss_per_epoch_list,color='g',label='Train_Loss')
    ax2.plot(range(len(val_avg_loss)),val_avg_loss,color='r',label='validation_Loss')

    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')

    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')

    ax1.set_title('Accuracy')
    ax2.set_title('Loss')

    ax1.legend()
    ax2.legend()

    plt.show()
#############################################


#load Model for isolation Forest
with open('iso_train.pickle', 'rb') as f:
    clf = pickle.load(f)

with open('svm_train.pickle', 'rb') as f:
    clf1 = pickle.load(f)
	
model = LetterCNN()
# load parameters for CNN
model.load_state_dict(torch.load('checkpoint.pt'))
# evaluate
model.eval()

# train_data=np.load("/content/drive/My Drive/ML_Data/data.npy",allow_pickle=True)
#train_data = load_pkl('/content/drive/My Drive/ML_Data/un_.pkl') 




###############################################################################################################################

# LOAD data
path = input("PLEASE ADD FILE(.pkl) PATH: ")
test_data = load_pkl(path) 
#test_data = np.load("data(1).npy",allow_pickle=True)
###########################################################################################################################





# !! if you want to take form as data[1] please use data[1:2] to keep way from error

predict_list=[]

image_tensor = preprocessing(test_data)
output=model(image_tensor)
# we ues the probabilities after softmax laber to train the Isolation Forest to rocognize unknown class.
pro8 = F.softmax(output,dim=1).data 


it=0
for i in range(len(test_data)):
  # if clf.predict(np.asarray(thin(stretch(test_data[i]))).flatten().reshape(1,32*32) )[0] == 1:
    #print(it)
    if clf.predict(np.array(pro8[i]).reshape(1,8))==-1 and clf1.predict(np.array(pro8[i]).reshape(1,8))==-1:
      predict_list.append(-1)
      print("Prediction is :",-1)
    else:
      _,predict = torch.max(pro8,1)
      predict = predict +1 
      predict_list.append(predict[i].item())
      
      print("Prediction is :",predict[i].item()) 
    it+=1
print("Final Predicts:",predict_list)

