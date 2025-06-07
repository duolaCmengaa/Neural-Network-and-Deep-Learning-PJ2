"""
Utils for neural networks
"""

from torch import nn
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

def count_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n

def setup_logging(model_name):
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'training_{model_name}.log')
    if not os.path.exists(log_filename):
        with open(log_filename, 'w'):
            pass
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_metrics(metrics, model_name):
    epochs = range(1, len(metrics['train_loss']) + 1)
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['test_loss'], label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['train_accuracy'], label='Train Accuracy', linewidth=1.5)
    plt.plot(epochs, metrics['test_accuracy'], label='Test Accuracy', linewidth=1.5)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['lr'], label='Learning Rate', linewidth=1.5)
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    
    os.makedirs('./figs', exist_ok=True)
    plt.savefig(f'./figs/{model_name}_metrics.png')
    plt.show()

def init_weights_(m):
    """
    Initializes weights of m according to Xavier normal method.

    :param m: module
    :return:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
