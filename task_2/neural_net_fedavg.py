import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784 #28x28
hidden_size = 500 #the number of nodes in the hidden layer
num_classes =10 #i.e. 10 digits here
num_epochs = 2 #these are the number of local epochs for each client
batch_size = 100
learning_rate = 0.001
num_clients = 10 
num_rounds = 5 #5 server-client-server communication rounds


#MNIST dataset

train_dataset = torchvision.datasets.MNIST(root = './data' , train = True , transform = transforms.ToTensor() , download = True)

test_dataset = torchvision.datasets.MNIST(root = './data' , train = False , transform = transforms.ToTensor())

# --- NEW: Distribute Data Among Clients ---
# We need to split the 60,000 training images among our 10 clients.
# We create a list of data indices for each client.
client_data_size = len(train_dataset) // num_clients
# Create a list of split lengths (e.g., [6000, 6000, ..., 6000])
split_lengths = [client_data_size] * num_clients
# Use random_split to divide the dataset
client_datasets = torch.utils.data.random_split(train_dataset, split_lengths)

# --- NEW: Create DataLoaders for each client ---
client_loaders = []
for client_ds in client_datasets:
    loader = torch.utils.data.DataLoader(dataset=client_ds, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)
    

#test cases loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self , input_size , hidden_size , num_classes):
        super(NeuralNet  , self).__init__()
        #defining the layers we want
        self.l1 = nn.Linear(input_size , hidden_size)
        #below is the activation function, we have used relu, we may also use softmax
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size , num_classes)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
    
#function for training each model on each client
def train_client(model , client_loader , num_epochs , learning_rate):

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #here we are using Adam for gradient calc, we may use SGD as well
    optimizer = torch.optim.Adam(model.parameters() , lr = learning_rate)

    #putting the model in training mode
    model.train()
    #training the model

    for epoch in range(num_epochs):
        for i, (image,labels) in enumerate(client_loader):
            #origin shape: [100,1,28,28] (the one here means the number of channels and 100 is the batch size)
            #resized: [100,784]
            #the -1 here means python has to determine the first dimension by itself 
            #the calculation is: 100*1*28*28 = 78400 elements in the original tensor
            #and the shape we asked was (?,784) , so ? becomes 78400/100 = 100
            #so the shape becomes (100,784)
            images = image.reshape(-1, 28*28).to(device)
            #labels are just the correct answers which we will compare our models with
            labels = labels.to(device)
            
            #forward pass and loss calc
            outputs = model(images)
            loss = criterion(outputs , labels)
            
            
            #backward and optimize
            loss.backward()
            optimizer.step()
            
            #setting the grad back to zero
            optimizer.zero_grad()
            
            return model.state_dict()
        

#FedAvg function
def federated_average(client_weights):
    # 'client_weights' is a list of state_dicts from the clients.
    # We will average them to get the new global model weights.
    
    # Get the keys (e.g., 'l1.weight', 'l1.bias') from the first model
    keys = client_weights[0].keys()
    new_global_weights = OrderedDict()
    
    for key in keys:
        # Stack all tensors for this key from all clients
        # e.g., stack all 'l1.weight' tensors
        key_tensors = torch.stack([weights[key] for weights in client_weights])
        
        # Calculate the mean along the 0-th dimension (the client dimension)
        key_mean = torch.mean(key_tensors, dim=0)
        
        # Store this averaged tensor in our new state_dict
        new_global_weights[key] = key_mean
        
    return new_global_weights          


#training the model

#making a global model
global_model = NeuralNet(input_size , hidden_size , num_classes).to(device)

print("--Starting Federated Training--")
for round in range(num_rounds):
    print(f"\n--Communication round : {round+1}/{num_rounds}--")
    
    #list to store model weights from each client
    client_weight_list = []
    
    #sending model to clients and training it
    for i in range(num_clients):
        # creating a copy of the global model for the client
        local_model = NeuralNet(input_size, hidden_size, num_classes).to(device)
        #setting its weights and biases to be same as the global model
        local_model.load_state_dict(global_model.state_dict())
        
        #getting the data loader for THIS client
        client_loader = client_loaders[i]
        
        
        #training the local model on client data
        print(f"Training Client {i+1}...")
        updated_weights = train_client(local_model ,client_loader,num_epochs , learning_rate )
        
        #collecting the updated weights
        client_weight_list.append(updated_weights)
    
    #aggregating and averaging the data from all the clients:
    print(f'Averaging client models...')
    new_global_weights = federated_average(client_weight_list)
    #updating the global model:
    global_model.load_state_dict(new_global_weights)
    
    #testing the model(we do not need to compute gradients, so we will turn it off)
    #we test after each round to see how it is improving
    with torch.no_grad():
        n_correct = 0;
        n_samples = len(test_loader.dataset)
        
        for images,labels in test_loader:
            images = images.reshape(-1,28*28).to(device)
            labels = labels.to(device)
            
            outputs = global_model(images)
            
            #max returns (output_value , index)
            _,predicted = torch.max(outputs,1)
            n_correct += (predicted == labels).sum().item()
            
        acc = n_correct/ n_samples
        print(f'Accuracy of the network on the {n_samples} test images: {100*acc:.4f}%')
        
    print('--Federated Training Complete--')
    
    
    