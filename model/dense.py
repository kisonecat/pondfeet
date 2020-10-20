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

        multiples = 10
        narrow = 6
        
        self.layer = nn.Sequential(
            nn.Linear(len(basicLetters) * width * height + 128 - 32,
                      multiples * width * height),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(multiples * width * height,
                      multiples * width * height),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(multiples * width * height,
                      narrow * width * height),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(narrow * width * height,
                      multiples * width * height),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(multiples * width * height,
                      multiples * width * height),
            nn.ReLU(),
            nn.Linear(multiples * width * height,
                      width * height)
            )

    def forward(self, code, inputs):

        code = self.embedding(code)
        out = inputs.reshape(-1, len(basicLetters) * width * height)
        out = torch.cat( (code, out), dim=1 )
        out = self.layer(out)
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
        code = code.to(device)
        
        # Forward pass
        outputs = model(code, inputs)
        loss = criterion(outputs, goal)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
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
torch.save(model.state_dict(), 'dense.ckpt')
