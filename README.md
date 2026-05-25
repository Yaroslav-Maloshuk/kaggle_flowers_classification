# Flowers Classification with ResNet18

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![torchvision](https://img.shields.io/badge/torchvision-models-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics-F7931E?logo=scikit-learn&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-notebook-F9AB00?logo=googlecolab&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-dataset-20BEFF?logo=kaggle&logoColor=white)

Transfer learning pipeline for 5-class flower recognition using a pretrained ResNet18 backbone with a fine-tuned classification head — trained on the Kaggle Flowers Dataset, achieving ~85% validation accuracy in 5 epochs.

## Why This Project

- Demonstrates transfer learning with frozen backbone + custom head
- Clean train/val split with full metric tracking per epoch
- Confusion matrix + F1 score for beyond-accuracy evaluation
- Prediction visualization with inverse normalization for human review

## Task

`Image → ResNet18 (frozen) → FC(512 → 5) → Flower Class`

## Technology Stack

- Model: `torchvision.models.resnet18` (pretrained on ImageNet)
- Training: PyTorch, CrossEntropyLoss, Adam optimizer
- Data: `datasets.ImageFolder`, `random_split`, `DataLoader`
- Metrics: scikit-learn (`confusion_matrix`, `f1_score`)
- Visualization: matplotlib, seaborn
- Dataset: Kaggle `imsparsh/flowers-dataset` via `kagglehub`

## Dataset

```text
flowers-dataset/
├── train/
│   ├── daisy/        501 images
│   ├── dandelion/    646 images
│   ├── rose/         497 images
│   ├── sunflower/    495 images
│   └── tulip/        607 images
└── test/             924 images (unlabeled)
```

80/20 train/val split applied to the train folder:

| Split      | Size  |
|------------|-------|
| Train      | 2 196 |
| Validation | 550   |
| Test       | 924   |

## Model Architecture

```text
ResNet18 (pretrained, all layers frozen)
    └── fc: Linear(512 → 5)   ← only this layer is trained
```

Only the final fully connected layer has `requires_grad = True`.

## Training Configuration

| Parameter     | Value              |
|---------------|--------------------|
| Epochs        | 5                  |
| Batch size    | 32                 |
| Optimizer     | Adam               |
| Learning rate | 0.001              |
| Loss          | CrossEntropyLoss   |
| Image size    | 150 × 150          |
| Device        | CUDA (if available)|

## Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 1.0593    | 60.34%    | 0.6457   | 80.73%  |
| 2     | 0.6039    | 79.37%    | 0.5036   | 84.00%  |
| 3     | 0.5063    | 82.33%    | 0.4655   | 86.18%  |
| 4     | 0.4752    | 84.47%    | 0.4643   | 84.00%  |
| 5     | 0.4324    | 85.29%    | 0.4302   | 85.45%  |

Best model checkpoint saved at epoch with lowest validation loss (`best_model.pth`).

## Notebook Structure

```text
notebook.ipynb
├── 1. Imports & Setup
├── 2. Dataset Download (kagglehub)
├── 3. Transforms & DataLoaders
│   ├── ImageFolder for train/val
│   └── CustomTestDataset for unlabeled test
├── 4. Model Definition (ResNet18 + custom head)
├── 5. Training Loop (5 epochs, best model saving)
├── 6. visualize_predictions()   — sample val images with true/pred labels
├── 7. plot_training_metrics()   — loss & accuracy curves
├── 8. Evaluation
│   ├── Validation loss, accuracy, F1 score
│   └── Confusion matrix (seaborn heatmap)
└── 9. Summary
```

## Quickstart (Google Colab)

```python
# 1. Install kagglehub if not present
!pip install kagglehub -q

# 2. Run all cells top to bottom
# All variables (model, val_loader, train_data) are defined in earlier cells

# 3. Visualize predictions
visualize_predictions(model, val_loader, train_data.classes, num_images=5)

# 4. Plot training curves
plot_training_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
```

## Summary

The ResNet18-based model successfully learned to recognize 5 classes of flowers and achieved an accuracy of approximately 85% on the validation set. Performance improved steadily during training with no significant overfitting. The model performs consistently across classes, with room for further improvement via unfreezing deeper layers, augmentation, or longer training.
