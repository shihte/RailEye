import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

def load_model(model_path):
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次維度
    
    # 檢測並使用可用的設備（MPS 或 CPU）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output)
        return probability.item()

# 使用範例
model = load_model('dog_model.pth')
result = predict_image(model, 'test_images/download.jpg')
print(f'是狗的機率: {result:.2%}')