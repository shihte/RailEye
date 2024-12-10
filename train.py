import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import time

# 訓練相關配置
TRAINING_EPOCHS = 70    # 訓練總輪數
BATCH_SIZE = 64        # 每批次處理的圖片數量
LEARNING_RATE = 0.001  # 學習率
MODEL_SAVE_PATH = 'dog_model.pth'  # 模型儲存路徑

# 數據增強參數
BRIGHTNESS = 0.2       # 亮度調整範圍
CONTRAST = 0.2         # 對比度調整範圍
ROTATION = 15          # 隨機旋轉角度範圍
IMAGE_SIZE = 224       # 輸入圖片尺寸

# 數據處理參數
NUM_WORKERS = 4        # 數據加載線程數

class DogDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 回傳浮點型標籤
        return image, torch.tensor(0.0, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer, num_epochs=TRAINING_EPOCHS):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)  # 扁平化輸出
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Loss: {running_loss/batch_count:.4f}')
        print(f'Time: {epoch_time:.2f}s')
        print('-' * 30)
    
    total_time = time.time() - start_time
    print(f'Total training time: {total_time/60:.2f} minutes')

def main():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ROTATION),
        transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data_dir = 'data'
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    dataset = DogDataset(image_paths, transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_model(model, train_loader, criterion, optimizer)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()