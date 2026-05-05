import torch
import torch.nn as nn
import timm

def get_convnext_nano_model(num_classes=19, pretrained=True, drop_rate=0.4, drop_path_rate=0.2):
    """
    Создает модель ConvNeXt Nano с кастомным количеством классов.
    """
    model = timm.create_model(
        'convnext_nano', 
        pretrained=pretrained, 
        num_classes=num_classes,
        drop_rate=drop_rate,       
        drop_path_rate=drop_path_rate
    )
    return model

if __name__ == "__main__":
    # Тестовый прогон
    model = get_convnext_nano_model(num_classes=19)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}") # Ожидаем [1, 19]
