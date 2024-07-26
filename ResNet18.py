import torch
import torch.nn as nn
import pandas as pd
import os
import re
import csv

from torch import optim
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import KFold

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        # 初始化shortcut连接
        # 如果stride!=1或in_channels!=out_channels，则使用1x1卷积来调整维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        # 通过shortcut连接获取与主路径输出相同维度的identity
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layer1 = self.make_layers(64, 64, 2, stride=1)
        self.res_layer2 = self.make_layers(64, 128, 2, stride=2)
        self.res_layer3 = self.make_layers(128, 256, 2, stride=2)
        self.res_layer4 = self.make_layers(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    # 用重复的残差块构建层的函数
    def make_layers(self, in_channels, out_channels, blocks_num, stride=1):
        layer = []
        layer.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks_num):
            layer.append(ResidualBlock(out_channels, out_channels, stride))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.res_layer1(out)
        out = self.res_layer2(out)
        out = self.res_layer3(out)
        out = self.res_layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


# 数据处理->得到一个图片集和一个标签的csv文件
# 1.处理xsl文件得到所有图片名的标签值列表，然后写入一个csv文件。
def get_labels_csv(data_path, label_xls):
    data_label = []
    label_xls_df = pd.read_excel(label_xls, header=0, index_col="姓名")
    data_dir = os.listdir(data_path)

    for img_name in data_dir:
        base_name = img_name.split(".")[0]
        person_name = re.sub(r'\d', '', base_name)
        # 使用正则去除数字,得到人名

        if person_name in label_xls_df.index:
            label = label_xls_df.loc[person_name, "表达方式"]
            data_label.append((img_name, label))
        else:
            print(f"{person_name} not found in the DataFrame.")

    # 把列表写入csv文件里
    label_csv = "image_label_csv"
    with open(label_csv, mode='w', encoding='utf-8') as f:
        # 创建一个csv.writer对象
        writer = csv.writer(f)
        writer.writerow(["img_name", "label"])
        # 写入每一行
        for img_name, label in data_label:
            writer.writerow([img_name, label])


# 重写dataset
class ImageDataset(Dataset):

    def __init__(self, data_dir, label_csv, transform):
        self.data_dir = data_dir
        self.label_df = pd.read_csv(label_csv)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["img_name"])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = row['label']
        return image, label

    def __len__(self):
        return len(self.label_df)


def train(model, train_loader, criterion, optimizer):
    model.train()
    batches_nums = 0
    running_acc = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # 变换labels的类型和形状与输出一样，可以用dtype和shape查看数据类型，形状
        labels = labels.type(torch.float).unsqueeze(1)
        # outputs是一个形状[batch_size,num_classes]的张量
        outputs = model(images)

        # 优化，计算loss
        loss_value = criterion(outputs, labels)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        batches_nums += 1
        # 计算准确率
        sigmoid = torch.sigmoid(outputs)
        # 将预测值转换为0，1
        predicted_classes = (sigmoid > 0.5).float()
        # sum把tensor所有元素求和，item把tensor转换为int类型,算出batch的对的个数
        correct = (predicted_classes == labels).sum().item()
        # 用正确个数除以总个数得到准确率
        batch_accuracy = correct / labels.size(0)
        running_acc += batch_accuracy

    avg_acc = running_acc / batches_nums
    return avg_acc


def test(model, test_loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_batches = 0

    with torch.no_grad():  # 测试时不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.type(torch.float).unsqueeze(1)
            outputs = model(images)
            # 计算loss
            loss_value = criterion(outputs, labels)
            total_loss += loss_value.item()

            # 计算准确率
            sigmoid = torch.sigmoid(outputs)
            # 将预测值转换为0，1
            predicted_classes = (sigmoid > 0.5).float()
            # sum把tensor所有元素求和，item把tensor转换为int类型,算出batch的对的个数
            correct = (predicted_classes == labels).sum().item()
            # 用正确个数除以总个数得到准确率
            acc_value = correct / labels.size(0)
            total_acc += acc_value

            total_batches += 1
    loss = total_loss / total_batches
    acc = total_acc / total_batches

    return loss, acc


data_path = "Cropped_images"
label_xls = "乳腺癌ki-67指数.xls"
# get_labels_csv(data_path, label_xls)
device = torch.device("cuda:{0}" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

epoch_nums = 10
kf = KFold(n_splits=5, shuffle=True)
x = pd.read_csv("image_label_csv")["img_name"]


for k, (train_idx, test_idx) in enumerate(kf.split(x)):
    dataset = ImageDataset(data_path, "image_label_csv", transform=transform)
    model = ResNet18(num_classes=1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # 适用于二分类问题的损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

    print(f"--------当前是第{k}折数据集------------")
    total_train_acc = 0
    test_acc = 0
    loos = 0
    for epoch in range(epoch_nums):
        train_acc = train(model, train_loader, criterion, optimizer)
        print(f'当前第{epoch}/{epoch_nums}个epoch的Avg_accuracy: {train_acc:.4f}')
        total_train_acc += train_acc

    avg_train_acc = total_train_acc / epoch_nums
    loss, test_acc = test(model, test_loader)

    print(f"第{k}折训练集的avg_train_acc为{avg_train_acc},测试集的loss, test_acc为：{loss}，{test_acc}")



