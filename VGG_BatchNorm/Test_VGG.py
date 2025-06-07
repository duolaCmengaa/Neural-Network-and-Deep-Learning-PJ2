import torch
from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

def get_accuracy(model, data_loader, is_train=True):
    if is_train == False:
        model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    if not is_train:
        print(f"Validation Accuracy: {accuracy:.4f} \n")
        model.train() 
    else:
        print(f"Train Accuracy: {accuracy:.4f} \n")
    return accuracy



def test_model(model_path, test_loader):

    # 注意替换为所要验证的模型类
    model = VGG_A()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_acc = get_accuracy(model, test_loader, is_train=False)
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/vgg_no_bn.pth"  
    val_loader, _ = get_cifar_loader(train=False)
    test_model(model_path, val_loader)
