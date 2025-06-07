
# Neural-Network-and-Deep-Learning-PJ2

## Project Description

The objective of this project is to:

1. Train neural network models on the CIFAR-10 dataset to optimize performance.  
2. Analyze how Batch Normalization (BN) aids optimization through experiments on the VGG-A architecture, with and without BN.

---

## First Task

First, navigate to the project directory:

```bash
cd My_nn_cifar10
````

To train the models, open `main.py` and run one of the following commands:

```bash
python main.py --model ResNet18
python main.py --model ResNet18_filtermul
python main.py --model ResNet18_dropout
python main.py --model ResNet34
```

If you'd like to explore the hyperparameter tuning process, refer to `Model_Tuning.ipynb`.

To visualize convolutional filters, see `Model_visualization.ipynb`.

To test the trained models:

```bash
python test_checkpoint.py
```

---

## Second Task

First, navigate to the second task directory:

```bash
cd VGG_BatchNorm
```

For Task 2.2: Run `VGG_Accuracy.py`.

For Task 2.3: Run `VGG_Loss_Landscape.py`.

To test the trained VGG models:

```bash
python Test_VGG.py
```



## Best Results Achieved

| Model               | Params (M) | Test Accuracy (%) | Avg. Epoch Time (s) |
| ------------------- | ---------- | ----------------- | ------------------- |
| ResNet18\_filtermul | 44.60      | 95.45             | 20.12               |
| ResNet34            | 21.28      | 95.45             | 14.67               |

