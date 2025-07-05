import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True, transform= transforms.ToTensor()), batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=1000, shuffle=False)

print(train_loader)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU() 

    def forward(self,x):
        x= x.view(-1,28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss,correct,total = 0.0,0,0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predicted = torch.max(output,1)
        if(batch_idx % 5000 == 0):
            print(f"Epoch {epoch+1} | Actual:    ", target[:20].tolist())
            print(f"Epoch {epoch+1} | Predicted: ", predicted[:20].tolist())
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(train_loader)
    
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")







