import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tkinter import Tk, Label, Button, filedialog, messagebox

# 定义与保存模型时相同的网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 加载模型
def load_model(model_path='handwritten_digit_recognition_model.pth'):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 评估文件夹
def evaluate_folder(model, folder_path):
    try:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        test_dataset = datasets.ImageFolder(root=folder_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        messagebox.showinfo("Accuracy", f"Accuracy on the dataset in {folder_path}: {accuracy:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Error evaluating folder: {e}")

# 预测单张图片
def predict_single_image(model, image_path):
    try:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        messagebox.showinfo("Prediction", f"Predicted label for {image_path}: {predicted.item()}")
    except Exception as e:
        messagebox.showerror("Error", f"Error predicting image: {e}")

# 主界面
class App:
    def __init__(self, master):
        self.master = master
        master.title("Handwritten Digit Recognition")

        self.label = Label(master, text="Choose an option below:")
        self.label.pack(pady=10)

        self.evaluate_folder_button = Button(master, text="Evaluate Folder", command=self.evaluate_folder)
        self.evaluate_folder_button.pack(pady=5)

        self.predict_image_button = Button(master, text="Predict Single Image", command=self.predict_image)
        self.predict_image_button.pack(pady=5)

        self.quit_button = Button(master, text="Quit", command=master.quit)
        self.quit_button.pack(pady=10)

        self.model = load_model()

    def evaluate_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            evaluate_folder(self.model, folder_path)

    def predict_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if image_path:
            predict_single_image(self.model, image_path)

# 启动应用
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
    
#pyinstaller --onefile --windowed digit_recognition_ui.py
