# CNN Stress-Testing: Fashion-MNIST Dataset

A comprehensive deep learning project that trains a CNN on Fashion-MNIST, identifies failure cases, and uses Grad-CAM explainability to analyze model predictions. The project compares a baseline model against a geometrically augmented model to demonstrate how data augmentation improves model robustness.

## 🎯 Project Overview

This project demonstrates:
- **Baseline CNN Training** on Fashion-MNIST dataset
- **Failure Case Analysis** identifying 6 distinct types of misclassifications
- **Grad-CAM Explainability** visualizing what the model focuses on
- **Data Augmentation** using geometric transforms to improve robustness
- **3-Panel Comparisons** showing baseline vs augmented model predictions

## 📊 Key Results

- **Baseline Model**: ~90% test accuracy
- **Augmented Model**: Improved robustness with geometric augmentation
- **6 Failure Cases Identified**:
  1. High-confidence misclassifications (>90% confidence, wrong)
  2. Shirt vs T-shirt confusion
  3. Coat vs Pullover confusion
  4. Sandal vs Sneaker confusion
  5. Dress vs Coat confusion
  6. Ankle Boot vs Sneaker confusion

## 🛠️ Requirements

### Dependencies

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
```

### Python Version
- Python 3.7 or higher

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/shyamaljoshi1/CNN-on-FashionMNIST
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python
```

## 🚀 How to Run

### Run in Jupyter Notebook

1. Install Jupyter:
```bash
pip install jupyter
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `main.ipynb` and run all cells sequentially.


## 🔬 Key Insights

### Baseline Model Issues
- **Overconfident errors**: Predicts wrong class with >90% confidence
- **Feature confusion**: Struggles with visually similar classes
- **Background bias**: Grad-CAM shows attention on irrelevant regions
- **Texture focus**: Prioritizes texture over shape

### Augmented Model Improvements
- **Better generalization**: Geometric transforms improve robustness
- **Shape-focused**: Grad-CAM shows improved attention on object shape
- **Reduced overfitting**: Better validation performance
- **Feature learning**: Learns more discriminative features

## 🔧 Customization

### Modify Hyperparameters
Edit these variables in the notebook:
```python
SEED = 42              # Random seed for reproducibility
batch_size = 128       # Batch size for training
num_epochs = 30        # Number of training epochs
learning_rate = 0.001  # Adam optimizer learning rate
```

### Adjust Data Augmentation
Modify augmentation strength:
```python
transforms.RandomRotation(10)  # Change rotation degree
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Change translation
transforms.RandomHorizontalFlip(p=0.5)  # Change flip probability
```

### Change CNN Architecture
Modify the `FashionCNN` class to experiment with:
- Number of convolutional layers
- Filter sizes and counts
- Fully connected layer sizes
- Dropout rates




