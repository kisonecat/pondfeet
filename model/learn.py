import torch 
import torch.nn as nn
import pickle
import random
import numpy as np
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
#num_epochs = 5
num_epochs = 1000
#batch_size = 100
batch_size = 1024
#learning_rate = 0.001
learning_rate = 0.001

basicLetters = 'MRAagbpyw'

print("Loading data...")
width = 6
height = 13
data = pickle.load(open("fonts.pkl","rb"))

train_dataset = []
test_dataset = []
for fontname in data.keys():
    glyphs = data[fontname]

    basicShapes = np.array([glyphs[ord(x)] for x in basicLetters])
    
    for c in glyphs.keys():
        if c not in [ord(x) for x in basicLetters]:
            glyph = c
        
            dataset = train_dataset
            if random.random() < 0.05:
                dataset = test_dataset

            row = (c - 32, basicShapes, glyphs[c])
            dataset.append(row)
    
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

print("Loaded!")
print("len(train_dataset) = ",len(train_dataset))
print("len(test_dataset) = ",len(test_dataset))

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.embedding = nn.Embedding(128-32,128-32)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(len(basicLetters), 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dropout1 = nn.Dropout(p=0.2)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dropout2 = nn.Dropout(p=0.2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(192 + 128 - 32, 100)

        self.fc2 = nn.Linear(100, height * width)

    def forward(self, code, inputs):

        code = self.embedding(code)
    
        out = self.layer1(inputs)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        
        out = out.reshape(-1, 192)
        out = torch.cat( (code, out), dim=1 )

        #print(inputs.size())
        #out = out.reshape(out.size(0), -1)
        #out = self.fc(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.reshape(-1, height, width)
        out = torch.sigmoid(out) # since we use BCELoss
        return out

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.BCELoss() # should be BCEWithLogitsLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (code, inputs, goal) in enumerate(train_loader):
        inputs = inputs.float().to(device)
        goal = goal.float().to(device)
        
        # Forward pass
        outputs = model(code, inputs)
        loss = criterion(outputs, goal)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (i+1) % 100 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for i, (code, inputs, goal) in enumerate(train_loader):    
        inputs = inputs.float().to(device)
        goal = goal.float().to(device)
        
        outputs = model(code, inputs)
        #print(code)
        #print(outputs)
        
        print()
        print()
        print(chr(code[0]+32))

        image = outputs[0]
        for y in range(height):
            for x in range(width):
                print(['.','░','▒','▓','█'][round(4*float(image[y,x]))],end='')
            print('')
        
        #_, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()

    #print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
