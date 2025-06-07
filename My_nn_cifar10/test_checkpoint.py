import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils.mynn import count_parameters

def get_testloader(batch_size=100, num_workers=4):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return testloader

def build_model(model_name):
    if model_name == 'ResNet18':
        return ResNet18()
    elif model_name == 'ResNet34':
        return ResNet34()
    elif model_name == 'ResNet50':
        return ResNet50()
    elif model_name == 'ResNet18_filtermul':
        return ResNet18_filtermul()
    elif model_name == 'ResNet18_dropout':
        return ResNet18_dropout()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return acc, avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet18', help='Model name')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model
    model = build_model(model_name).to(device)

    # 加载权重
    checkpoint_path = f'./checkpoints/{model_name}_ckpt.pth'
    assert os.path.isfile(checkpoint_path), f'Checkpoint not found: {checkpoint_path}'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])

    # 模型参数信息
    total_params = count_parameters(model)
    print(f'Model: {model_name}, Total Parameters: {total_params}')

    # 加载测试数据
    testloader = get_testloader()

    # 测试模型
    acc, avg_loss = test_model(model, testloader, device)
    print(f'[Test] Accuracy: {acc:.2f}%, Avg Loss: {avg_loss:.4f}')

if __name__ == '__main__':
    main()
