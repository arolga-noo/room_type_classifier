import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloaders import create_dataloaders  # Используем системный загрузчик проекта
from models.convnext_nano.model import get_convnext_nano_model
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Загрузка данных (пути подтянутся автоматически из структуры проекта)
train_loader, val_loader = create_dataloaders(
    batch_size=32, 
    image_size=224,
    use_weighted_sampling=False # Эвристики уже выровняли баланс
)

# 2. Инициализация модели
model = get_convnext_nano_model(num_classes=19).to(DEVICE)

# 3. Настройки обучения
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)

# 4. Простейший цикл обучения
print("Начало обучения ConvNeXt Nano...")
model.train()
for epoch in range(5):
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} completed.")

# Сохранение весов
torch.save(model.state_dict(), "models/convnext_nano/best_model.pth")
