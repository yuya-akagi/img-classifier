import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import random_split

# データ前処理
transform = transforms.Compose([
    transforms.Resize(224),   # GoogleNet用にサイズを変更
    transforms.Grayscale(3),  # グレースケールをRGBに変換
    transforms.ToTensor(),    # テンソル化
    transforms.Normalize((0.5,), (0.5,))  # 標準化
])

# MNISTデータセットをダウンロードしてロード
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# トレーニングデータを80%/20%に分割して検証データを作成
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# データローダーを作成
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 事前学習されたGoogleNetをロード
model = models.googlenet(pretrained=True)

# 出力層をMNISTの10クラスに変更
model.fc = nn.Linear(1024, 10)  # GoogleNetの最終層は1024ユニットなので、それを10に変更

# GPUが使用可能であれば、モデルをGPUに移動
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数とオプティマイザの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニング関数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

# 学習中のテスト関数 (train_val: 学習中に実施する教師データありのテスト)
def train_val(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy

# 学習後のテスト関数 (validation: 学習後に実施する教師データありのテスト)
def validation(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# 学習後の最終テスト (test: 教師データなしのテスト)
def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print(f'Predicted: {predicted}')  # デモ用に予測結果を出力

# 学習ループ関数
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_save_path):
    best_val_accuracy = 0.0  # ベストな検証精度を保存するための変数
    
    for epoch in range(num_epochs):
        # トレーニング
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # 学習中の検証
        val_loss, val_accuracy = train_val(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # 検証精度が最高のとき、モデルを保存
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved at Epoch {epoch+1} with Val Accuracy: {val_accuracy:.2f}%')

    # 学習後のテストで評価 (教師データあり)
    final_val_accuracy = validation(model, test_loader, criterion, device)
    print(f'Final Validation Accuracy (with labels): {final_val_accuracy:.2f}%')
    
    # ラベルなしの最終テスト
    test(model, test_loader, device)

# 学習を実行する
num_epochs = 5
model_save_path = 'googlenet_mnist_best.pth'  # モデルの保存先
train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, model_save_path)
