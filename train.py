import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import logging
import datetime
from src.dataloader import *
from src.model import LeNet

LOG_DIR = "logs"
MODELS_DIR = "saved_models"
FIG_DIR = "figures"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(LOG_DIR, f"train_log_{current_time}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

model = LeNet()
model.to(DEVICE)

batch_size = 64
lr = 0.001
epochs = 30
criterion = nn.CrossEntropyLoss()
optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


train_loader, test_loader = get_mnist_datas(batch_size)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def test_epoch(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            prediction = torch.argmax(outputs, dim=1)
            
            correct += (prediction == labels).sum().item()
            
            total += labels.size(0)

    return correct / total


def train(model, train_loader, test_loader, epochs, criterion, optimizer, device):
    loss_his = []
    acc_his = []
    best_acc = 0
    logging.info("------ Training Started ------\n")
    for epoch in range(epochs):
        logging.info(f"--- Epoch [{epoch+1}/{epochs}] ---")

        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        acc = test_epoch(model, test_loader, device)

        loss_his.append(loss)
        acc_his.append(acc)

        logging.info(f"    Average Loss: {loss:.4f}")
        logging.info(f"    Test Accuracy: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model_path = os.path.join(MODELS_DIR, "mnist_net_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"        -> New best model saved!")

        

    logging.info("\n------ Training Finished ------")
    logging.info("Saving the final model.....")
    last_model_path = os.path.join(MODELS_DIR, "mnist_net_last.pth")
    torch.save(model.state_dict(), last_model_path)
    logging.info(f"Final model saved to: {last_model_path}")

    return loss_his, acc_his

def draw_figures(loss_his, acc_his, epochs):
    logging.info("\nPlotting and saving curves.....")
    epochs_range = range(1, epochs+1)

    plt.figure()
    plt.title("Training Loss")
    plt.plot(epochs_range, loss_his)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    loss_path = os.path.join(FIG_DIR, "loss_curve.png")
    plt.savefig(loss_path)
    logging.info(f"Loss curve saved to {loss_path}")

    plt.figure()
    plt.title("Accuracy")
    plt.plot(epochs_range, acc_his)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    acc_path = os.path.join(FIG_DIR, "acc_curve.png")
    plt.savefig(acc_path)
    logging.info(f"Accuracy curve saved to {acc_path}")

loss_his, acc_his = train(model, train_loader, test_loader, epochs, criterion, optimizer, DEVICE)

draw_figures(loss_his, acc_his, epochs)