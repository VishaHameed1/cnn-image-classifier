import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

# ✅ Paths to your real dataset
train_dir = r"C:\Users\as\Documents\ai-ml-beginner-journey\intermediate-ml\cnn-image-classifier\PetImages"
classes = ['Cat', 'Dog']  # folder names in PetImages

# 1️⃣ Dataset transforms (with augmentation)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# 2️⃣ Prepare dataset (skip corrupted images)
all_images = []
all_labels = []

for idx, cls in enumerate(classes):
    cls_folder = os.path.join(train_dir, cls)
    for file in os.listdir(cls_folder):
        file_path = os.path.join(cls_folder, file)
        if file.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                # Try to open image to filter out corrupted ones
                img = Image.open(file_path)
                img.verify()  # Will raise error if image is corrupted
                all_images.append(file_path)
                all_labels.append(idx)
            except (UnidentifiedImageError, OSError):
                print(f"Skipped corrupted image: {file_path}")

# 80% train, 20% test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# 3️⃣ Custom Dataset
class PetDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')  # Ensure 3 channels
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_data = PetDataset(train_paths, train_labels, transform=transform)
test_data  = PetDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=32)

# 4️⃣ Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# 5️⃣ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Training loop
for epoch in range(10):  # more epochs for real data
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# 7️⃣ Testing accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")