import torch
from PIL import Image
from torchvision import transforms
import argparse
from src.model import LeNet

parser = argparse.ArgumentParser(description="Predict a handwritten digit")

parser.add_argument(
    '--image_path', 
    type=str,
    required=True,
    help='Path to the image'
)
parser.add_argument(
    '--model_path',
    type=str,
    default='saved_models/mnist_net_best.pth',
    help='Path to the trained .pth model file'
)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Image to predict: {args.image_path}")
print(f"Model file used: {args.model_path}")

model = LeNet()
saved_weights = torch.load(args.model_path)
model.load_state_dict(saved_weights)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x)
])

image = Image.open(args.image_path)
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to(DEVICE)

print("\nPredicting...")
with torch.no_grad():
    output = model(input_batch)
    print(f"Raw model output (Logits):\n{output}\n")

    predicted_index = torch.argmax(output, dim=1)
    prediction = predicted_index.item()

print(f"-> The predicted digit is: {prediction}")