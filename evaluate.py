import torch
from torchvision import transforms
import PIL.Image as Image
import matplotlib.pyplot as plt
from efficientnet.model import EfficientNet
from data.transforms import val_transform

model_name = 'b0'
num_classes = 37
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = EfficientNet(model_name, num_classes)
model.load_state_dict(torch.load('efficientnet_pet_model.pth'))
model = model.to(device)
model.eval()

# Load and preprocess a sample image
sample_image_path = 'example.jpg' # Replace with the path to your sample image
sample_image = Image.open(sample_image_path).convert('RGB')

transform = val_transform
input_tensor = transform(sample_image).unsqueeze(0).to(device)

# Run the model on the sample image
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

# Map predicted class index to class name
from torchvision.datasets import OxfordIIITPet
train_dataset = OxfordIIITPet(root='./data', split='trainval', download=True)
class_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_idx.items()}
predicted_class = idx_to_class[predicted.item()]

print(f'Predicted Class: {predicted_class}')

# Display the image and prediction
predicted_label = predicted_class
plt.imshow(sample_image)
plt.title(f'Predicted: {predicted_label}')
plt.show()
