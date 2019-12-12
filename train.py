import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import UrbanSound8KDataset

# Prepare arrays for curve plotting at the end
LMC_train_loss = []
LMC_test_loss = []
LMC_train_accuracy = []
LMC_test_accuracy = []

MLMC_train_loss = []
MLMC_test_loss = []
MLMC_train_accuracy = []
MLMC_test_accuracy = []

MC_train_loss = []
MC_test_loss = []
MC_train_accuracy = []
MC_test_accuracy = []

# Initialise summary writer
summary_writer = SummaryWriter('logs',flush_secs=5)

# CUDA
print("cuda is available: ", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load data
#region data
train_loader_LMC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'),
    batch_size=32, shuffle=True,
    num_workers=8, pin_memory=True)

test_loader_LMC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)

train_loader_MC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MC'),
    batch_size=32, shuffle=True,
    num_workers=8, pin_memory=True)

test_loader_MC = torch.utils.data.DataLoader(
    UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MC'),
    batch_size=32, shuffle=False,
    num_workers=8, pin_memory=True)

train_loader_MLMC = torch.utils.data.DataLoader(
      UrbanSound8KDataset('UrbanSound8K_train.pkl', 'MLMC'),
      batch_size=32, shuffle=True,
      num_workers=8, pin_memory=True)

test_loader_MLMC = torch.utils.data.DataLoader(
     UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MLMC'),
     batch_size=32, shuffle=False,
     num_workers=8, pin_memory=True)


#endregion data

# Define the network classes
#region NetworkClasses
class LMC_Net(nn.Module):
    #Initialisation method
    def __init__(self):
        super().__init__()

        ##1st layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3,3),
            padding = 1 
        )

        self.norm1 = nn.BatchNorm2d(num_features=32)
        
        ##2nd layer
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (3,3),
            dilation=2,
            padding = 1
        )

        self.norm2 = nn.BatchNorm2d(num_features = 32)

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),padding=1)

        ##3rd layer
        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3,3),
            dilation=2,
            padding = 1
        )

        self.norm3 = nn.BatchNorm2d(num_features = 64)

        ##4th layer
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (2,2),  
            dilation=2,
            padding = 1
        )
        
        self.norm4 = nn.BatchNorm2d(num_features = 64)

        ##5th layer
        # self.fc1 = nn.Linear(in_features=15488,out_features=1024)#no dilation
        self.fc1 = nn.Linear(in_features=46080,out_features=1024)#dilation of 2,2,2
        
        #6th layer
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024 ,out_features=10)

    def forward(self,x):
        ##1
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        ##2
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.dropout1(x)

        ##3
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        ##4
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.dropout2(x)

        #Flatten
        x = torch.flatten(x,start_dim = 1)

        ##5
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout3(x)

        ##6
        x = self.fc2(x)

        return x

class MLMC_Net(nn.Module):
    #Initialisation method
    def __init__(self):
        super().__init__()
        #define layers here

       ##1st layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3,3),
            padding = 1 
        )

        self.norm1 = nn.BatchNorm2d(num_features=32)
        

        ##2nd layer
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 32,
            kernel_size = (3,3),
            dilation=2,
            padding = 1 
        )

        self.norm2 = nn.BatchNorm2d(num_features = 32)

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),padding=1)

        ##3rd layer
        self.conv3 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3,3),
            dilation=2,
            padding = 1
        )

        self.norm3 = nn.BatchNorm2d(num_features = 64)

        ##4th layer
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv4 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = (3,3),
            stride = (2,2),
            dilation=2,
            padding = 1
        )
        
        self.norm4 = nn.BatchNorm2d(num_features = 64)

        ##5th layer
        # self.fc1 = nn.Linear(in_features=26048,out_features=1024)#no dilation
        self.fc1 = nn.Linear(in_features=80640,out_features=1024)#dilation of 2,2,2
        
        #6th layer
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024 ,out_features=10)

    def forward(self,x):
        ##1
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)

        ##2
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        ##3
        x = self.conv3(x)

        x = self.norm3(x)
        x = F.relu(x)

        ##4
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = self.dropout2(x)

        #Flatten
        x = torch.flatten(x,start_dim = 1)

        ##5
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout3(x)

        ##6
        x = self.fc2(x)

        return x

#endregion NetworkClasses


# Accuracy function
def accuracy(logits, targets):
    correct = (torch.argmax(logits,1) == targets).sum()
    accuracy = float(correct)/targets.shape[0]
    return accuracy

# Training/ testing function
def trainAndValidate(model, 
                    trainingData,
                    testData, 
                    logitFilenameDictionary,
                    targetFilenameDictionary,
                    tensorboardDatasetName,
                    numEpochs=8, 
                    learningRate=0.001, 
                    # momentum_=0.9, 
                    weightDecay=1e-5
                    ):
    print("Training and validating ", tensorboardDatasetName)
    print("Learning rate:          ", learningRate)
    print("numEpochs:              ", numEpochs)
    print("weightDecay:            ", weightDecay)

    # Define optimiser
    optimiser = optim.Adam(
        params=model.parameters(),
        lr=learningRate,
        weight_decay=weightDecay
        )

    criterion = nn.CrossEntropyLoss()

    #####EPOCH LOOP#######
    for epoch in range(0,numEpochs):
        print(epoch)

        #(Only used for printing)
        myLoss = 0
        myAcc = 0

        #####TRAINING LOOP#######
        model.train()

        #for each batch (input is 32 images)
        for i,(input,target,filenames) in enumerate(trainingData):
            #training loop for single batch
            #Get input and target
            input = input.to(device)
            target = target.to(device)

            # Compute logits
            logits = model(input)

            # Compute loss
            loss = criterion(logits,target)
            myLoss = loss.item()

            # Backprop/step
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            # Compute acc
            train_accuracy = accuracy(logits, target)*100
            
            ########################matPlotLib###########################################
            if(tensorboardDatasetName == "MC"):
                MC_train_accuracy.append(train_accuracy)
                MC_train_loss.append(loss.item())

            if(tensorboardDatasetName == "LMC"):
                LMC_train_accuracy.append(train_accuracy)
                LMC_train_loss.append(loss.item())

            if(tensorboardDatasetName == "MLMC"):
                MLMC_train_accuracy.append(train_accuracy)
                MLMC_train_loss.append(loss.item())
            #############################################################################

            # Add to summary writer
            summary_writer.add_scalar(('loss/train-'+tensorboardDatasetName), loss.item(), epoch)   ##per batch training loss
            summary_writer.add_scalar(('accuracy/train-'+tensorboardDatasetName), train_accuracy, epoch)  ##per batch training accuracy
            myAcc = train_accuracy

        print(myLoss)
        print(myAcc)

        #####TESTING LOOP#######
        # Turn off dropout and batchnorm layers
        model.eval()

        #####TEST LOOP#######
        # Don't need to track grad
        with torch.no_grad():

            numTestBatch = len(testData)
            totalLoss    = 0
            
            softmax = nn.Softmax(dim=0)
            # For each batch in test set
            for i,(input,target,filenames) in enumerate(testData):
                input  = input.to(device)   
                target = target.to(device)
                logits = model(input)
                loss = criterion(logits,target) 
                totalLoss += loss.item()

                #Construct two dictionaries: 
                #   logitFilenameDictionary:  map filename -> all logits for this filename, 
                #   targetFilenameDictionary: map filename -> target
                
                #for each image
                for j in range(0,len(filenames)):
                    #check if image's filename already exists in the logit dictionary as a key
                    if filenames[j] in logitFilenameDictionary:
                        # if it does: append logits (one list of 10 values) for this image
                        logitFilenameDictionary[filenames[j]].append(logits[j])
                    else:
                        #otherwise, create new key, and append these logits
                        logitFilenameDictionary[filenames[j]] = []
                        logitFilenameDictionary[filenames[j]].append(logits[j])

                    #check if image's filename already exists in the target dictionary
                    if filenames[j] not in targetFilenameDictionary:
                        #associate filename to target
                        targetFilenameDictionary[filenames[j]] = target[j]
                
            #average test loss for this epoch
            averageLoss = float(totalLoss) / float(numTestBatch)
            print("numtestbatch",numTestBatch)
            print("totalloss  ",totalLoss)
            print("averageloss", averageLoss)
            summary_writer.add_scalar(('loss/test-'+tensorboardDatasetName), averageLoss, epoch)  #per epoch test loss

            #calculating accuracy using the dictionaries
            correctPredsPerClass = torch.zeros(10).to(device)
            noFilesPerClass = torch.zeros(10).to(device)
            # count the number of files per class
            for keyFilename in targetFilenameDictionary:
                #add up number of filenames in each class
                noFilesPerClass[targetFilenameDictionary[keyFilename]] += 1

            #Test accuracy for this epoch
            #For each filename in the dictionary
            for filename in logitFilenameDictionary:
                logitsList = logitFilenameDictionary[filename] #all logits for this filename (for the clips corresponding to this file)
                
                #next few lines are to sum the logits for this filename, so that argmax can be called
                #logits sum is the elementwise sum of logits 
                logitsSum = torch.zeros(10).to(device)
                for logits in logitsList:
                    logitsSum += softmax(logits)
                
                #if the overall prediction (based on the summed logits) for this file is correct, increment correct predictions    
                if(logitsSum.argmax(dim=-1)) == targetFilenameDictionary[filename]:
                    correctPredsPerClass[targetFilenameDictionary[filename]] += 1
            
            #test accuracy is obtained by dividing the number of correctly identified files, by the number of files
            testAccPerClass = torch.div(correctPredsPerClass, noFilesPerClass)
            print("Per test accuracy of",tensorboardDatasetName,"=",testAccPerClass)
            aveAcc = (torch.sum(testAccPerClass) / 10)
            print("Average accuracy of",tensorboardDatasetName,"=",aveAcc)

            summary_writer.add_scalar(('accuracy/test-'+tensorboardDatasetName), aveAcc, epoch)  ##per epoch test accuracy

            ########################matPlotLib###########################################
            if(tensorboardDatasetName == "MC"):
                MC_test_accuracy.append(aveAcc)
                MC_test_loss.append(averageLoss)

            if(tensorboardDatasetName == "LMC"):
                LMC_test_accuracy.append(aveAcc)
                LMC_test_loss.append(averageLoss)

            if(tensorboardDatasetName == "MLMC"):
                MLMC_test_accuracy.append(aveAcc)
                MLMC_test_loss.append(averageLoss)
            #############################################################################

    summary_writer.close()

# Helpful function for printing model params
def PrintModelParameters(model):
    # print(LMC_model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size())


# Training/ testing models
#region Models

############### LMC ###############
###The dictionaries get filled by running train and validate, and then these dictionaries are used for TSCNN
LMC_logitFilenameDictionary = {}
LMC_targetFilenameDictionary = {}
LMC_model = LMC_Net().to(device)
trainAndValidate(LMC_model, train_loader_LMC, test_loader_LMC, LMC_logitFilenameDictionary, LMC_targetFilenameDictionary, 'LMC', 50, 0.001, 1e-5)

############### MC ###############
MC_logitFilenameDictionary = {}
MC_targetFilenameDictionary = {}
MC_model = LMC_Net().to(device)  #MC_Model has identical architecture to LMC_Model, wo we instantiate the same network class
trainAndValidate(MC_model, train_loader_MC, test_loader_MC, MC_logitFilenameDictionary, MC_targetFilenameDictionary, 'MC', 50, 0.001, 1e-5)

############### TSCNN ###############
def TSCNN():
    softmax = nn.Softmax(dim=0)
    #LMC Probs
    for filename in LMC_logitFilenameDictionary:
        logitsList = LMC_logitFilenameDictionary[filename] #all logits for this filename (for the clips corresponding to this file)
        
        #next few lines are to sum the logits for this filename, so that argmax can be called
        #logits sum is the elementwise sum of logits 
        logitsSum = torch.zeros(10).to(device)
        for logits in logitsList:
            logitsSum += softmax(logits)

        #Store new probs in dictionary
        LMC_logitFilenameDictionary[filename] = logitsSum

    #MC Probs
    for filename in MC_logitFilenameDictionary:
        logitsList = MC_logitFilenameDictionary[filename] #all logits for this filename (for the clips corresponding to this file)
        
        #next few lines are to sum the logits for this filename, so that argmax can be called
        #logits sum is the elementwise sum of logits 
        logitsSum = torch.zeros(10).to(device)
        for logits in logitsList:
            logitsSum += softmax(logits)

        #Store new probs in dictionary
        MC_logitFilenameDictionary[filename] = logitsSum

    correctPredsPerClass = torch.zeros(10).to(device)
    noFilesPerClass = torch.zeros(10).to(device)
    #combination of probs
    for filename in LMC_logitFilenameDictionary:
        #combine and argmax
        pred = torch.argmax(softmax(MC_logitFilenameDictionary[filename]) + softmax(LMC_logitFilenameDictionary[filename]), dim=-1)

        #count number of files per class
        noFilesPerClass[LMC_targetFilenameDictionary[filename]] += 1
        
        #check for correct prediction
        if (pred == LMC_targetFilenameDictionary[filename]):
            #incr. class
            correctPredsPerClass[LMC_targetFilenameDictionary[filename]] += 1

    testAccPerClass = torch.div(correctPredsPerClass, noFilesPerClass)

    print("Accuracy per class for TSCNN:",testAccPerClass)
    averageAccuracy = (torch.sum(testAccPerClass) / 10)
    print()
    print("Average accuracy for TSCNN:",averageAccuracy.item())


# Call TSCNN
TSCNN()


############### MLMC ###############
MLMC_logitFilenameDictionary = {}
MLMC_targetFilenameDictionary = {}
MLMC_model = MLMC_Net().to(device)
trainAndValidate(MLMC_model, train_loader_MLMC, test_loader_MLMC, MLMC_logitFilenameDictionary,MLMC_targetFilenameDictionary, 'MLMC',50, 0.001, 1e-5)


# Save all three models
torch.save(LMC_model.state_dict() , "models/lmc.pt")
torch.save(MC_model.state_dict()  , "models/mc.pt")
torch.save(MLMC_model.state_dict(), "models/mlmc.pt")


#endregion Models


# Plot and output all curves with MatPlotLib
#region Plotting
# LMC:
plt.plot(LMC_train_loss)
plt.savefig("figs/LMC_train_loss")
plt.clf()
plt.plot(LMC_test_loss)
plt.savefig("figs/LMC_test_loss")
plt.clf()
plt.plot(LMC_train_accuracy)
plt.savefig("figs/LMC_train_accuracy")
plt.clf()
plt.plot(LMC_test_accuracy) 
plt.savefig("figs/LMC_test_accuracy")
plt.clf()

# MLMC
plt.plot(MLMC_train_loss)
plt.savefig("figs/MLMC_train_loss")
plt.clf()
plt.plot(MLMC_test_loss)
plt.savefig("figs/MLMC_test_loss")
plt.clf()
plt.plot(MLMC_train_accuracy)
plt.savefig("figs/MLMC_train_accuracy")
plt.clf()
plt.plot(MLMC_test_accuracy)
plt.savefig("figs/MLMC_test_accuracy")
plt.clf()

# MC
plt.plot(MC_train_loss)
plt.savefig("figs/MC_train_loss")
plt.clf()
plt.plot(MC_test_loss)
plt.savefig("figs/MC_test_loss")
plt.clf()
plt.plot(MC_train_accuracy)
plt.savefig("figs/MC_train_accuracy")
plt.clf()
plt.plot(MC_test_accuracy)
plt.savefig("figs/MC_test_accuracy")
plt.clf()
#endregion Plotting
