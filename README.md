# ResNet for MNIST

An educational implementation of a residual convolutional neural network in PyTorch. The project demonstrates how identity and projection shortcuts help information and gradients flow through a deep network, using handwritten-digit classification on MNIST as the example task.

## What is a residual network?

A conventional stack of layers tries to learn a direct mapping, `H(x)`. A residual block instead learns a residual function, `F(x) = H(x) - x`, and combines it with the original input:

```text
y = F(x) + x
```

```text
             ┌── convolution → batch norm → ReLU
x ───────────┤
             └── shortcut ───────────────────────→ add → output
```

The shortcut is an identity connection when the input and output shapes match. When a block changes the number of channels or spatial resolution, this implementation uses a `1 × 1` convolution to project the shortcut to the required shape.

Residual connections make very deep networks easier to optimize by giving gradients a direct path through the network. They also let a block approximate an identity mapping when additional transformations are not useful.

## Current architecture

The network in [`model/resnet.py`](model/resnet.py) accepts grayscale MNIST images and produces logits for 10 digit classes.

| Stage            | Main path                                                   | Shortcut                              | Output channels |
| ---------------- | ----------------------------------------------------------- | ------------------------------------- | --------------: |
| Stem             | `7 × 7` convolution, batch normalization, ReLU, max pooling | —                                     |              64 |
| Residual stage 1 | Two residual blocks with `3 × 3` convolutions               | Identity                              |              64 |
| Residual stage 2 | Two residual blocks with `3 × 3` convolutions               | `1 × 1` projection in the first block |             128 |
| Residual stage 3 | Two residual blocks with `3 × 3` convolutions               | `1 × 1` projection in the first block |             256 |
| Classifier       | Global average pooling and a fully connected layer          | —                                     |              10 |

The script currently uses these training settings:

| Setting       |         Value |
| ------------- | ------------: |
| Dataset       |         MNIST |
| Batch size    |           100 |
| Epochs        |             2 |
| Optimizer     |          Adam |
| Learning rate |          0.01 |
| Loss          | Cross-entropy |

## Repository structure

```text
ResNet/
├── data/
│   └── MNIST/          # Downloaded automatically when the script runs
├── model/
│   └── resnet.py       # Model definition, training loop, and evaluation loop
└── README.md
```

## Requirements

- Python 3.9 or newer
- PyTorch
- torchvision
- NumPy
- Matplotlib

NumPy and Matplotlib are imported by the current script but are not yet used in the training workflow.

## Setup

Clone the repository and enter its directory:

```bash
git clone <repository-url>
cd ResNet
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install torch torchvision numpy matplotlib
```

For platform-specific PyTorch installation commands, especially when using an NVIDIA GPU, consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Running the project

From the repository root, run:

```bash
python model/resnet.py
```

The script downloads MNIST into `data/` when necessary, creates the model, trains it, and evaluates classification accuracy on the test split. During training it prints the loss every 100 mini-batches; after evaluation it prints test accuracy.

Device selection is automatic in the current code:

- Apple Metal (`mps`) is used when available.
- CPU is used otherwise.
- CUDA is not currently selected automatically.

Hyperparameters are defined near the top of `model/resnet.py` and can be edited directly.

## References

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015.
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision MNIST dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
