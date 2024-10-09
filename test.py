import torch
from torchvision import transforms
from PIL import Image
from model import ConvLSTMModel

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMModel().to(device)
model.load_state_dict(torch.load('model_weights.pth'))  # 加载训练好的模型权重
model.eval()

# 定义图片转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 测试函数
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    with torch.no_grad():
        outputs = model(image)
        angle, torque, speed = outputs.cpu().numpy()[0]
        print(f"Predicted Angle: {angle:.4f}, Torque: {torque:.4f}, Speed: {speed:.4f}")


# 测试单张图片
test_image = 'you image path'
predict(test_image)
