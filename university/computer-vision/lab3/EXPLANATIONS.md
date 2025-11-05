# Lab 3: Deep Dive Explanations

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [DenseNet Architecture](#densenet-architecture)
4. [Training DenseNet](#training-densenet)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Siamese Networks](#siamese-networks)
7. [Triplet Loss](#triplet-loss)
8. [Similar Image Search](#similar-image-search)
9. [t-SNE Visualization](#t-sne-visualization)
10. [Comparison Analysis](#comparison-analysis)

---

## Overview

This lab implements three major deep learning concepts:

1. **DenseNet** - A convolutional neural network with dense connections for classification
2. **Siamese Networks** - Twin networks that learn similarity metrics
3. **t-SNE** - Dimensionality reduction for visualizing high-dimensional embeddings

The goal is to classify Fashion-MNIST images with >90% accuracy, then use learned representations to find similar images.

---

## Data Preparation

### Fashion-MNIST Dataset

Fashion-MNIST is a dataset of 70,000 grayscale images (28×28 pixels) across 10 clothing categories:

- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Training set**: 60,000 images
- **Test set**: 10,000 images

### Data Augmentation

```python
transforms.RandomRotation(10)
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
```

**Why augmentation?**

- **Prevents overfitting**: Creates variations of training data
- **Improves generalization**: Model learns to recognize patterns regardless of small rotations/shifts
- **RandomRotation(10)**: Rotates images ±10 degrees (clothing can be worn at slight angles)
- **RandomAffine translate=(0.1, 0.1)**: Shifts images up to 10% in x/y directions (objects not always centered)

### Normalization

```python
transforms.Normalize((0.5,), (0.5,))
```

**Purpose**: Transforms pixel values from [0, 1] to [-1, 1]

- **Formula**: `normalized = (x - mean) / std = (x - 0.5) / 0.5`
- **Benefits**:
  - Centers data around zero → faster convergence
  - Reduces internal covariate shift
  - Helps with gradient flow in deep networks

### Train/Val/Test Split

- **Train (90%)**: Used for learning weights
- **Validation (10%)**: Used for hyperparameter tuning and early stopping
- **Test**: Held out for final evaluation (no training decisions based on this)

**Why separate validation?**

- Test set must remain untouched to avoid "leaking" information
- Validation set helps detect overfitting during training

---

## DenseNet Architecture

### What is DenseNet?

DenseNet (Densely Connected Convolutional Networks) introduced a revolutionary idea: **connect each layer to every subsequent layer**.

**Key Innovation**: Instead of traditional sequential connections (layer 1 → 2 → 3), DenseNet uses:

```
Layer 1 → Layer 2, 3, 4, ...
Layer 2 → Layer 3, 4, ...
Layer 3 → Layer 4, ...
```

### Why DenseNet Works

1. **Feature Reuse**: Early layers learn low-level features (edges, textures) which are reused by later layers
2. **Gradient Flow**: Direct connections create shorter paths for gradients during backpropagation → alleviates vanishing gradient problem
3. **Parameter Efficiency**: Fewer parameters than ResNet for similar performance
4. **Implicit Deep Supervision**: Each layer receives supervision from the loss function through multiple paths

### Architecture Components

#### 1. Dense Layer (Bottleneck Design)

```python
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout=0.2):
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1)  # Bottleneck
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
```

**Bottleneck Architecture (BN-ReLU-Conv)**:

- **1×1 convolution**: Reduces computational cost

  - Input: `in_channels` → Output: `4 * growth_rate` (typically 4k)
  - **Why?** 3×3 convolutions are expensive; 1×1 reduces dimensions first
  - Example: 128 channels → 48 channels → much cheaper 3×3 conv

- **3×3 convolution**: Learns spatial features

  - Output: `growth_rate` new feature maps (typically k=12)
  - **padding=1**: Maintains spatial dimensions

- **Batch Normalization**:

  - Normalizes activations to mean=0, std=1
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Acts as regularization

- **Dropout(0.2)**:
  - Randomly drops 20% of activations during training
  - Prevents co-adaptation of neurons
  - Regularization technique

**Forward Pass**:

```python
def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    out = self.dropout(out)
    return torch.cat([x, out], 1)  # Concatenate input with output
```

**Critical**: `torch.cat([x, out], 1)` concatenates input with new features

- Input: 128 channels
- New features: 12 channels (growth_rate)
- Output: 140 channels (128 + 12)

#### 2. Dense Block

```python
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])
```

**Progressive Channel Growth**:

- Layer 1: `in_channels` → `in_channels + growth_rate`
- Layer 2: `in_channels + growth_rate` → `in_channels + 2*growth_rate`
- Layer 3: `in_channels + 2*growth_rate` → `in_channels + 3*growth_rate`
- ...
- Layer n: `in_channels + (n-1)*growth_rate` → `in_channels + n*growth_rate`

**Example with growth_rate=12, num_layers=6**:

- Input: 24 channels
- After layer 1: 36 channels (24 + 12)
- After layer 2: 48 channels (36 + 12)
- After layer 6: 96 channels (24 + 72)

**Growth Rate (k)**: Controls model capacity

- Small k (12): Fewer parameters, faster training
- Large k (32): More expressive, higher capacity
- Trade-off: memory vs. performance

#### 3. Transition Layer

```python
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(2, stride=2)
```

**Purpose**: Reduce spatial dimensions and control channel growth

**Compression Factor**: `θ = out_channels / in_channels` (typically 0.5)

- If Dense Block output: 200 channels
- Transition output: 100 channels (0.5 × 200)
- **Benefit**: Prevents explosive channel growth, reduces parameters

**Average Pooling (2×2)**:

- Reduces spatial dimensions by half: 28×28 → 14×14 → 7×7
- **Why Average vs Max?**: Smoother downsampling, better gradient flow

#### 4. Complete DenseNet Structure

```
Input: 1×28×28 (grayscale)
    ↓
Initial Conv: 1×28×28 → 24×28×28
    ↓
Dense Block 1 (6 layers): 24×28×28 → 96×28×28
    ↓
Transition 1: 96×28×28 → 48×14×14  (compression=0.5)
    ↓
Dense Block 2 (12 layers): 48×14×14 → 192×14×14
    ↓
Transition 2: 192×14×14 → 96×7×7
    ↓
Dense Block 3 (8 layers): 96×7×7 → 192×7×7
    ↓
Batch Norm + ReLU
    ↓
Global Average Pooling: 192×7×7 → 192×1×1
    ↓
Fully Connected: 192 → 10 classes
```

**Parameter Calculation Example**:

- Dense Block: Each layer adds k feature maps, n layers → k×n new features
- Total: Much fewer parameters than traditional CNNs due to feature reuse

### Weight Initialization

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)  # He initialization
```

**Kaiming (He) Initialization**:

- Designed for ReLU activations
- Sets weights from normal distribution with variance: `2 / n_in`
- **Why?** Maintains variance of activations across layers
- Prevents vanishing/exploding gradients at initialization

---

## Training DenseNet

### Loss Function: Cross-Entropy Loss

```python
criterion = nn.CrossEntropyLoss()
```

**Mathematical Definition**:

```
L = -Σ y_true * log(y_pred)
```

For classification with one-hot encoded labels:

```
L = -log(p_correct_class)
```

**Why Cross-Entropy?**

- **Probabilistic interpretation**: Maximizes likelihood of correct class
- **Penalizes confidence in wrong answers**: High penalty for confident wrong predictions
- **Gradient properties**: Well-behaved gradients for SGD

**Example**:

- True class: 2 (Pullover)
- Predictions: [0.1, 0.05, 0.7, 0.05, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005]
- Loss: -log(0.7) = 0.357 (low loss, good prediction)
- If prediction was 0.1: -log(0.1) = 2.303 (high loss, bad prediction)

### Optimizer: Adam

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

**Adam (Adaptive Moment Estimation)**:

- Combines momentum and RMSprop
- Maintains per-parameter learning rates
- Adaptive to gradient magnitude

**Key Components**:

1. **Momentum** (β₁=0.9):

   - Smooths gradient updates
   - Accelerates convergence
   - `m_t = β₁ * m_{t-1} + (1-β₁) * g_t`

2. **RMSprop** (β₂=0.999):

   - Adapts learning rate per parameter
   - Uses exponential moving average of squared gradients
   - `v_t = β₂ * v_{t-1} + (1-β₂) * g_t²`

3. **Update Rule**:
   ```
   m̂_t = m_t / (1 - β₁^t)  # Bias correction
   v̂_t = v_t / (1 - β₂^t)
   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
   ```

**Weight Decay (L2 Regularization)**:

- `weight_decay=1e-4` adds penalty term: `λ * ||W||²`
- Prevents large weights → reduces overfitting
- Encourages simpler models

### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
```

**ReduceLROnPlateau Strategy**:

- **mode='max'**: Monitors validation accuracy (we want to maximize)
- **factor=0.5**: Reduces LR by half when plateau detected
- **patience=3**: Waits 3 epochs before reducing LR

**Why Dynamic LR?**

- **Initial phase**: High LR (0.001) for fast convergence
- **Fine-tuning**: Low LR (0.0005 → 0.00025) for refinement
- **Plateau detection**: If accuracy doesn't improve for 3 epochs → reduce LR
- Helps escape local minima and find better solutions

**Example Schedule**:

```
Epochs 1-10:   LR = 0.001  (accuracy improving)
Epochs 11-13:  LR = 0.001  (plateau, accuracy ~89%)
Epoch 14:      LR = 0.0005 (reduced, accuracy jumps to 90%)
Epochs 15-18:  LR = 0.0005 (further improvement to 91%)
```

### Training Loop

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # Enable dropout, batch norm training mode

    for images, labels in loader:
        optimizer.zero_grad()      # Clear gradients from previous batch
        outputs = model(images)    # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()            # Backpropagation
        optimizer.step()           # Update weights
```

**Key Steps**:

1. **model.train()**:

   - Enables dropout (randomly drops neurons)
   - Batch norm uses batch statistics (not running averages)

2. **optimizer.zero_grad()**:

   - PyTorch accumulates gradients by default
   - Must clear before each backward pass

3. **Forward Pass**:

   - Input flows through network
   - Computes predictions

4. **Loss Computation**:

   - Measures error between predictions and true labels

5. **Backward Pass (loss.backward())**:

   - Computes gradients via automatic differentiation
   - Chain rule applied through all layers

6. **Weight Update (optimizer.step())**:
   - Updates parameters using computed gradients
   - Adam applies momentum and adaptive learning rates

### Validation Loop

```python
def validate(model, loader, criterion, device):
    model.eval()  # Disable dropout, use running stats for batch norm

    with torch.no_grad():  # Disable gradient computation
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
```

**Key Differences from Training**:

1. **model.eval()**:

   - Disables dropout → all neurons active
   - Batch norm uses running statistics (stable, learned during training)

2. **torch.no_grad()**:

   - Disables gradient computation → saves memory
   - Faster inference (no need to store intermediate activations)

3. **No Optimization**:
   - No backward pass
   - No weight updates
   - Only evaluating performance

### Early Stopping

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'best_densenet.pth')
```

**Concept**: Save model when validation performance improves

**Why?**

- Model might overfit after peak performance
- Training acc keeps improving, but val acc decreases
- Early stopping saves the generalization sweet spot

**Example**:

```
Epoch 15: Val Acc = 90.5% → Save model
Epoch 16: Val Acc = 90.3% → Don't save
Epoch 17: Val Acc = 90.8% → Save model (new best)
Epoch 25: Val Acc = 89.5% → Don't save (overfitting started)
```

---

## Evaluation Metrics

### Confusion Matrix

```python
cm = confusion_matrix(all_labels, all_preds)
```

**What it shows**:

- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

**Example**:

```
           T-shirt  Trouser  Pullover  ...
T-shirt      950      5        30      ...
Trouser       2     985        1       ...
Pullover     25      0       960       ...
```

**Insights**:

- T-shirt often confused with Pullover (similar shape)
- Trouser rarely confused (distinctive shape)
- Confusion reveals semantic similarity

### Classification Report

```python
classification_report(all_labels, all_preds, target_names=class_names)
```

**Metrics per class**:

1. **Precision**: Of all predicted X, how many were actually X?

   ```
   Precision = TP / (TP + FP)
   ```

   - High precision → low false positive rate
   - Example: If precision=0.95 for Sneaker, 95% of predicted sneakers are correct

2. **Recall**: Of all actual X, how many did we find?

   ```
   Recall = TP / (TP + FN)
   ```

   - High recall → low false negative rate
   - Example: If recall=0.90 for Sneaker, we found 90% of all sneakers

3. **F1-Score**: Harmonic mean of precision and recall
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```
   - Balances precision and recall
   - Useful when you care about both metrics equally

**Trade-off Example**:

- **High Precision, Low Recall**: Conservative classifier (few false alarms, but misses many)
- **Low Precision, High Recall**: Liberal classifier (finds everything, but many false alarms)

**Support**: Number of samples in each class (helps assess reliability)

---

## Siamese Networks

### Concept

**Traditional Classification**: Learn to map images → class labels

```
Image → CNN → [0.1, 0.05, 0.7, ...] → Class
```

**Siamese Networks**: Learn to map images → embedding space where similar items are close

```
Image₁ → CNN → Embedding₁ ──┐
                             ├── Distance
Image₂ → CNN → Embedding₂ ──┘
```

### Why Siamese Networks?

1. **Few-Shot Learning**: Can compare with new classes not in training set
2. **Similarity Learning**: Learns what makes items similar (not just classification)
3. **Embedding Quality**: Creates meaningful representation space
4. **Verification Tasks**: "Are these two images the same person/object?"

### Architecture

```python
class SiameseNetwork(nn.Module):
    def __init__(self, backbone, embedding_dim=128):
        self.backbone = backbone  # Pre-trained DenseNet
        self.embedding = nn.Sequential(
            nn.Linear(backbone.num_features, embedding_dim),  # 192 → 128
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)           # 128 → 128
        )
```

**Design Choices**:

1. **Backbone (DenseNet)**:

   - Pre-trained on classification task
   - Already learned useful features (edges, textures, shapes)
   - **Transfer learning**: Reuse knowledge from classification

2. **Embedding Layer**:

   - Maps DenseNet features (192-dim) → compact embeddings (128-dim)
   - **Dimensionality reduction**: Removes redundant information
   - **Non-linear transformation**: Two FC layers with ReLU (more expressive)

3. **L2 Normalization**:
   ```python
   return F.normalize(embeddings, p=2, dim=1)
   ```
   - Projects embeddings onto unit hypersphere
   - **Why?** Makes distance comparisons fair (all vectors same magnitude)
   - Changes from Euclidean distance to cosine similarity

### Freezing Backbone

```python
for param in siamese_model.backbone.parameters():
    param.requires_grad = False
```

**Why Freeze?**

- **Faster training**: Only update embedding layer (~33k params vs ~500k params)
- **Stability**: Backbone already learned good features
- **Prevents catastrophic forgetting**: Don't destroy classification features

**Alternative**: Fine-tune entire network with very low learning rate

---

## Triplet Loss

### Concept

**Goal**: Learn embeddings where:

- **Similar images** (same class) are close together
- **Dissimilar images** (different classes) are far apart

**Triplet**: Three images per training sample

1. **Anchor**: Reference image
2. **Positive**: Same class as anchor
3. **Negative**: Different class from anchor

### Mathematical Definition

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)  # ||a - p||²
        neg_dist = F.pairwise_distance(anchor, negative, p=2)  # ||a - n||²
        loss = F.relu(pos_dist - neg_dist + margin)            # max(0, ...)
        return loss.mean()
```

**Formula**:

```
L = max(0, ||a - p||² - ||a - n||² + margin)
```

**Interpretation**:

- `||a - p||²`: Distance between anchor and positive (want this SMALL)
- `||a - n||²`: Distance between anchor and negative (want this LARGE)
- `margin`: Minimum separation between positive and negative

**Loss Behavior**:

1. **Case 1**: `pos_dist=0.2, neg_dist=0.8, margin=0.5`

   ```
   loss = max(0, 0.2 - 0.8 + 0.5) = max(0, -0.1) = 0
   ```

   - Already separated by more than margin → no update needed

2. **Case 2**: `pos_dist=0.6, neg_dist=0.7, margin=0.5`

   ```
   loss = max(0, 0.6 - 0.7 + 0.5) = max(0, 0.4) = 0.4
   ```

   - Not separated enough → push negative further

3. **Case 3**: `pos_dist=0.5, neg_dist=0.3, margin=0.5`
   ```
   loss = max(0, 0.5 - 0.3 + 0.5) = max(0, 0.7) = 0.7
   ```
   - Negative closer than positive → large penalty

### Margin Selection

**Margin=0.2**: Tight constraint

- Forces very clear separation
- May be too strict (hard to optimize)

**Margin=0.5**: Moderate constraint (our choice)

- Good balance between separation and optimization difficulty

**Margin=1.0**: Loose constraint

- Easier to optimize initially
- May not create tight clusters

**Tuning Strategy**: Start with larger margin, decrease as training progresses

### Triplet Mining

**Random Triplets** (our approach):

```python
positive_idx = np.random.choice(self.label_to_indices[anchor_label])
negative_label = np.random.choice([l for l != anchor_label])
negative_idx = np.random.choice(self.label_to_indices[negative_label])
```

**Problem**: Many triplets are "easy" (loss=0)

- Most random negatives are already far from anchor
- Network doesn't learn from easy examples

**Advanced: Hard Negative Mining**:

1. **Hard Negative**: Negative sample closest to anchor
2. **Semi-Hard Negative**: Negative farther than positive, but within margin
3. **Online Mining**: Select hardest samples within each batch

**Benefits of Hard Mining**:

- Faster convergence (only train on informative examples)
- Better embeddings (focuses on decision boundaries)
- More efficient training

---

## Similar Image Search

### Embedding Space

After Siamese training, each image maps to 128-dimensional embedding:

```
Image → [e₁, e₂, e₃, ..., e₁₂₈]
```

**Properties of Good Embeddings**:

1. **Intra-class similarity**: Images of same class cluster together
2. **Inter-class separation**: Different classes far apart
3. **Semantic meaning**: Distance reflects visual similarity

### Distance Metrics

#### Euclidean Distance (L2)

```python
distances = np.linalg.norm(embeddings - query_embedding, axis=1)
```

**Formula**:

```
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

**Properties**:

- Standard distance metric
- Sensitive to magnitude
- Works well with L2-normalized embeddings

#### Cosine Similarity

```python
similarity = np.dot(embeddings, query_embedding.T) / (
    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
)
```

**Formula**:

```
sim(x, y) = (x · y) / (||x|| * ||y||)
```

**Properties**:

- Measures angle between vectors (ignores magnitude)
- Range: [-1, 1] where 1=identical direction, -1=opposite
- Better for high-dimensional spaces
- With L2-normalized embeddings, equivalent to dot product

**Why L2 Normalization?**

```python
F.normalize(embeddings, p=2, dim=1)  # Normalize to unit length
```

- Makes all embedding vectors have length=1
- Euclidean distance ≈ Cosine similarity for unit vectors
- Simplifies distance computation
- Removes bias from magnitude differences

### k-Nearest Neighbors (k-NN)

```python
def find_similar_images(query_idx, embeddings, top_k=5):
    query_embedding = embeddings[query_idx]
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    similar_indices = np.argsort(distances)[1:top_k+1]  # Skip first (self)
    return similar_indices
```

**Algorithm**:

1. Compute distance from query to all other embeddings
2. Sort by distance (ascending)
3. Return k closest neighbors (excluding query itself)

**Complexity**: O(n\*d) where n=dataset size, d=embedding dimension

- For 10,000 images × 128 dims: ~1.28M operations (very fast)

**Advanced: Efficient Search**:

- **FAISS**: Facebook's library for billion-scale similarity search
- **Approximate NN**: Trade accuracy for speed (LSH, HNSW)
- **Indexing structures**: KD-trees, Ball trees

### Evaluation Metrics

#### Top-k Accuracy

```python
top_k_labels = labels[sorted_indices[:k]]
accuracy = np.mean(top_k_labels == query_label)
```

**Definition**: Fraction of k nearest neighbors with same class as query

**Example**: Query is "Sneaker"

- Top-5 results: [Sneaker, Sneaker, Ankle boot, Sneaker, Sneaker]
- Top-5 accuracy: 4/5 = 80%

**Why Top-k?**

- Top-1 might be too strict (some classes very similar)
- Top-5 or Top-10 gives more realistic retrieval performance
- Reflects real-world use case (show user several options)

#### Mean Average Precision (mAP)

**More sophisticated metric** that considers:

1. **Precision at each rank**: What % of retrieved images are relevant?
2. **Average Precision**: Average of precision values at ranks where relevant items appear
3. **Mean**: Average AP across all queries

**Formula**:

```
AP = (1/R) * Σ P(k) * rel(k)
```

where:

- R = total relevant items
- P(k) = precision at rank k
- rel(k) = 1 if item at rank k is relevant, 0 otherwise

**Example**: Query "Sneaker", Top-10 results

```
Ranks:   1    2    3    4    5    6    7    8    9   10
Class:   Snk  Snk  Ank  Snk  Bag  Snk  Snk  Ank  Snk  Snk
Rel:     1    1    0    1    0    1    1    0    1    1
Prec:   1.0  1.0  0.67 0.75 0.6  0.67 0.71 0.63 0.67 0.70
```

AP = (1.0 + 1.0 + 0.75 + 0.67 + 0.71 + 0.67 + 0.70) / 7 = 0.79

**Why mAP?**

- Considers order (finding relevant items early is better)
- Balances precision and recall
- Standard metric in information retrieval

---

## t-SNE Visualization

### What is t-SNE?

**t-Distributed Stochastic Neighbor Embedding**:

- Non-linear dimensionality reduction technique
- Maps high-dimensional data (128D) to 2D/3D for visualization
- Preserves local structure (nearby points stay nearby)

### How t-SNE Works

#### Step 1: Compute Pairwise Similarities (High-Dimensional)

For each pair of points (i, j), compute conditional probability:

```
p(j|i) = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σₖ exp(-||xᵢ - xₖ||² / 2σᵢ²)
```

**Interpretation**: Probability that point i would pick j as neighbor

**Perplexity**: Controls effective number of neighbors

- Low perplexity (5-15): Focus on very local structure
- Medium perplexity (30-50): Balance local and global
- High perplexity (100+): More global structure

#### Step 2: Compute Similarities (Low-Dimensional)

In 2D embedding space, use Student's t-distribution (heavy-tailed):

```
q(j|i) = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖ (1 + ||yᵢ - yₖ||²)⁻¹
```

**Why t-distribution?**

- Heavy tails allow moderate distances in high-D to map to larger distances in 2D
- Alleviates "crowding problem" (all points can't fit in 2D)

#### Step 3: Minimize KL Divergence

Objective: Make q distributions match p distributions

```
KL(P || Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)
```

**Optimization**: Gradient descent on embedding coordinates

```
∂KL/∂yᵢ = 4 Σⱼ (pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹
```

### Hyperparameters

```python
TSNE(n_components=2, perplexity=40, learning_rate=200, n_iter=1000)
```

#### Perplexity (30-50)

**Definition**: Smooth measure of effective number of neighbors

**Effects**:

- **Low (5-15)**: Many small, tight clusters (may miss global structure)
- **Medium (30-50)**: Good balance (our choice)
- **High (100+)**: Emphasizes global structure (may merge distinct clusters)

**Rule of thumb**: Perplexity = 5-50 for datasets of 1k-10k samples

#### Learning Rate (200)

**Effects**:

- **Too low (10-50)**: Slow convergence, may get stuck
- **Good (100-500)**: Balanced optimization
- **Too high (1000+)**: Unstable, clusters may not form

**Adaptive**: t-SNE uses momentum-based optimization

#### Number of Iterations (1000+)

**Typical needs**:

- Small datasets: 250-500 iterations
- Medium datasets: 1000 iterations (our choice)
- Large datasets: 2000+ iterations

**How to tell if enough?**

- Plot KL divergence vs iteration
- Should converge to stable value
- More iterations = better convergence (but diminishing returns)

### PCA Pre-reduction

```python
pca = PCA(n_components=50)
embeddings_pca = pca.fit_transform(sample_embeddings)
```

**Why PCA first?**

1. **Speed**: t-SNE is O(n² log n) → very slow for high dimensions

   - 128D × 5000 samples: ~10 minutes
   - 50D × 5000 samples: ~2 minutes

2. **Noise reduction**: PCA removes least important dimensions

   - First 50 PCs typically capture >95% variance
   - Removes noise in tail dimensions

3. **Better results**: Less noise → cleaner clusters

**PCA vs t-SNE**:

- **PCA**: Linear, preserves global structure, fast
- **t-SNE**: Non-linear, preserves local structure, slow but more expressive

### Interpreting t-SNE Plots

**Good Embedding Space**:

```
✓ Clear class clusters (same-class points grouped)
✓ Separation between classes (different classes apart)
✓ Semantic proximity (similar classes nearby)
  - Sneaker and Ankle boot close together (both footwear)
  - T-shirt, Pullover, Shirt close together (upper body wear)
```

**Poor Embedding Space**:

```
✗ Mixed clusters (classes overlap heavily)
✗ No clear structure (random scatter)
✗ Outliers everywhere (poor generalization)
```

**Common Patterns**:

- **Satellite clusters**: Subcategories within class (e.g., different sneaker styles)
- **Overlapping classes**: Semantically similar (Pullover vs Shirt)
- **Isolated clusters**: Distinctive classes (Bag, very different from clothing)

### Limitations of t-SNE

1. **Non-deterministic**: Different runs give different layouts (but similar structure)
2. **Distance interpretation**: Distances between clusters not meaningful (only within-cluster)
3. **Slow**: Not suitable for real-time or very large datasets
4. **Hyperparameter sensitive**: Must tune perplexity, learning rate

**Alternatives**:

- **UMAP**: Faster, preserves global structure better
- **PaCMAP**: Combines PCA and t-SNE benefits

---

## Comparison Analysis

### Why Compare Three Representations?

Understanding how learned representations improve over raw data:

1. **Raw Pixels** (784-D): Each pixel is a feature
2. **DenseNet Features** (192-D): Learned features before classification
3. **Siamese Embeddings** (128-D): Optimized for similarity

### Raw Pixel t-SNE

**Expected Result**: Poor separation

```
Challenges:
- High dimensionality (784 features)
- Pixel values don't capture semantics
- Small variations (shift, rotation) cause large distance changes
- No learned structure
```

**What it shows**: Some clustering by color/texture, but classes overlap heavily

### DenseNet Feature t-SNE

**Expected Result**: Good separation

```
Advantages over raw pixels:
- Learned hierarchical features (edges → textures → parts → objects)
- Translation invariant (due to convolutions)
- Dimensionality reduced (192-D)
- Optimized for classification task
```

**What it shows**: Clear class clusters, but optimized for classification (not similarity)

### Siamese Embedding t-SNE

**Expected Result**: Best separation

```
Advantages over DenseNet:
- Explicitly trained for similarity (triplet loss)
- Further dimensionality reduction (128-D)
- L2 normalized (distance-aware)
- Metric learning optimizes embedding space
```

**What it shows**: Tightest clusters, best inter-class separation, semantic similarity preserved

### Quantitative Comparison

**Cluster Quality Metrics**:

1. **Silhouette Score**: Measures cluster cohesion and separation

   ```
   s(i) = (b(i) - a(i)) / max(a(i), b(i))
   ```

   - a(i) = avg distance to same-cluster points
   - b(i) = avg distance to nearest different-cluster points
   - Range: [-1, 1], higher is better

2. **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances

   - Lower is better
   - Measures cluster separation

3. **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
   - Higher is better

**Expected Results**:

```
                    Raw Pixels  DenseNet  Siamese
Silhouette Score:      0.15       0.35      0.52
Davies-Bouldin:        2.8        1.4       0.8
Calinski-Harabasz:     450        1200      2100
```

### Visual Analysis

**Overlap Analysis**:

- Count points from different classes within each cluster
- Measure cluster purity: % of dominant class in each cluster

**Semantic Grouping**:

```
Expected semantic groups in Siamese embeddings:
1. Footwear: Sneaker, Ankle boot, Sandal
2. Upper body: T-shirt, Pullover, Shirt, Coat
3. Lower body: Trouser
4. Accessories: Bag
5. Formal wear: Dress
```

**Progressive Improvement**:

- Raw → DenseNet: Learn features instead of pixels
- DenseNet → Siamese: Optimize for similarity instead of classification

---

## Key Takeaways

### DenseNet Architecture

✓ Dense connections enable feature reuse and gradient flow
✓ Bottleneck layers reduce computational cost
✓ Transition layers control model complexity
✓ Achieves >90% accuracy with fewer parameters than traditional CNNs

### Siamese Networks

✓ Learn similarity metrics, not just classification
✓ Transfer learning from classification task accelerates training
✓ Triplet loss creates semantic embedding space
✓ Enable few-shot learning and similarity search

### Metric Learning

✓ Embeddings capture semantic similarity
✓ Distance in embedding space reflects visual similarity
✓ L2 normalization standardizes distance comparisons
✓ Hard negative mining improves efficiency

### t-SNE Visualization

✓ Non-linear dimensionality reduction preserves local structure
✓ Perplexity controls local vs global focus
✓ PCA pre-reduction speeds up computation
✓ Visual inspection reveals embedding quality

### Comparison Insights

✓ Learned representations vastly superior to raw pixels
✓ Task-specific training (Siamese) outperforms classification features
✓ Dimensionality reduction doesn't hurt when done right
✓ Visualization confirms quantitative metrics

---

## Further Reading

### Papers

1. **DenseNet**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (Huang et al., 2017)
2. **Siamese Networks**: [Learning a Similarity Metric Discriminatively](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf) (Chopra et al., 2005)
3. **Triplet Loss**: [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832) (Schroff et al., 2015)
4. **t-SNE**: [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) (van der Maaten & Hinton, 2008)

### Concepts

- **Metric Learning**: Learning distance functions for similarity
- **Few-Shot Learning**: Learning from few examples per class
- **Transfer Learning**: Reusing learned features for new tasks
- **Embedding Spaces**: Learned representations for downstream tasks

### Extensions

1. **Online Triplet Mining**: Mine hard examples during training
2. **Multi-task Learning**: Combine classification + similarity losses
3. **Attention Mechanisms**: Focus on relevant image regions
4. **Contrastive Learning**: Self-supervised learning (SimCLR, MoCo)
5. **UMAP**: Alternative to t-SNE for visualization

---

## Frequently Asked Questions

### 1. Яка основна ідея використання Siamese мереж для пошуку подібних зображень?

**Ключова ідея**: Siamese мережі навчають **метричний простір вкладень** (embedding space), де візуально подібні зображення розташовані близько, а різні - далеко. На відміну від класифікації, яка просто призначає мітку класу, Siamese мережі вивчають **функцію подібності**, яка узагальнюється на нові об'єкти поза навчальним набором. Це досягається через triplet loss: `L = max(0, d(anchor, positive) - d(anchor, negative) + margin)`, який змушує мережу зменшувати відстань між anchor і positive (той самий клас) і збільшувати відстань до negative (інший клас).

### 2. Яким чином генеруються батчі для навчання мережі?

**Triplet Dataset генерує батчі з трійок (anchor, positive, negative)**:

- **Anchor**: Випадкове зображення з датасету
- **Positive**: Інше зображення того ж класу (не те саме зображення)
- **Negative**: Випадкове зображення з іншого класу

Реалізація: `label_to_indices` словник створює індекси для кожного класу, дозволяючи швидкий семплінг. Для кожного anchor випадково вибираємо positive з того ж класу та negative з іншого класу. **Hard negative mining** (не реалізовано в базовій версії) вибирає найскладніші негативи для швидшої конвергенції.

### 3. Як перевірити, що генератор батчів працює коректно?

**Методи валідації**:

1. **Візуальна перевірка**: Вивести кілька трійок і перевірити, що positive - той самий клас, negative - інший
2. **Статистична перевірка**: Підрахувати розподіл класів у positive/negative парах (має бути балансований)
3. **Тест міток**: `assert positive_label == anchor_label and negative_label != anchor_label`
4. **Дистанції**: До тренування перевірити, що d(anchor, positive) і d(anchor, negative) рівномірно розподілені (після тренування перша має бути меншою)

### 4. Як створити конволюційну нейронну мережу для генерації фічей з зображень?

**Архітектура Feature Extraction CNN** (наш DenseNet):

1. **Початковий conv layer**: Перетворює вхідні зображення в feature maps
2. **Dense blocks**: Послідовні шари з dense connections для вивчення ієрархічних ознак (edges → textures → parts → objects)
3. **Transition layers**: Зменшують просторові розміри (pooling) і контролюють ріст каналів (compression)
4. **Global Average Pooling**: Перетворює feature maps в вектор ознак (192-D у нашому випадку)
5. **Без останнього FC layer**: Для feature extraction видаляємо класифікаційний шар, використовуючи `get_features()` метод

**Ключові принципи**: Batch normalization (стабілізація), dropout (regularization), ReLU activations, Kaiming initialization.

### 5. Які етапи необхідні для компіляції та оптимізації Siamese моделі?

**Етапи**:

1. **Backbone заморозка**: `param.requires_grad = False` для DenseNet (використовуємо pretrained features)
2. **Embedding layer**: Додаємо FC layers (192 → 128 → 128) для метричного навчання
3. **L2 нормалізація**: `F.normalize()` проектує embeddings на одиничну гіперсферу
4. **Triplet Loss**: Реалізуємо з margin (0.5) для розділення класів
5. **Optimizer**: Adam з низьким LR (0.001) тільки для embedding layer
6. **Scheduler**: StepLR для зменшення LR кожні 5 epochs
7. **Валідація**: Моніторинг triplet loss та intra/inter-class distances

### 6. Які переваги використання CNN на основі LeNet-5 для класифікації зображень?

**LeNet-5 переваги** (хоча ми використали DenseNet):

- **Простота**: Лише 2 conv + 2 FC layers - легко зрозуміти і дебажити
- **Швидкість**: Малий розмір (~60k параметрів) - швидке тренування/інференс
- **Baseline**: Добрий початковий експеримент для нових задач
- **Малі датасети**: Працює для простих задач (MNIST, Fashion-MNIST 28×28)

**Обмеження**: Недостатньо глибокий для складних задач; сучасні архітектури (DenseNet, ResNet) значно кращі на складних датасетах.

### 7. Як налаштувати модель CNN на основі AlexNet для досягнення необхідної точності?

**Стратегії налаштування** (застосовні до будь-якої CNN):

1. **Data augmentation**: RandomRotation, RandomAffine, ColorJitter (збільшує різноманітність даних)
2. **Learning rate**: Почати з 0.001, використати scheduler (ReduceLROnPlateau або CosineAnnealing)
3. **Regularization**: Dropout (0.2-0.5), weight decay (1e-4), batch normalization
4. **Batch size**: 64-256 (більший batch = стабільніший градієнт, але потребує більше пам'яті)
5. **Optimizer**: Adam (адаптивний LR) або SGD+momentum (потребує більше тюнінгу, але може дати кращі результати)
6. **Architecture tuning**: Збільшити/зменшити кількість filters, додати/видалити layers
7. **Early stopping**: Зберігати найкращу модель за validation accuracy

### 8. Які архітектури CNN підходять для класифікації зображень в датасеті fashion-mnist?

**Рекомендовані архітектури** (для 28×28 grayscale):

1. **DenseNet** (наш вибір): Dense connections, ефективний використання параметрів, >90% accuracy
2. **ResNet-18**: Residual connections, глибока мережа, відмінна точність
3. **MobileNet**: Легка архітектура, швидка інференс, добре для mobile/edge devices
4. **EfficientNet-B0**: Балансує точність і ефективність, state-of-the-art результати
5. **VGG-16** (спрощена): Проста архітектура, але багато параметрів
6. **Custom small CNN**: 3-4 conv layers + pooling + FC, якщо потрібна простота

**Критерії вибору**: Точність потрібна (>90%?), швидкість інференсу, розмір моделі, доступні ресурси (GPU memory).

### 9. Як перевірити корисність фічей, згенерованих моделлю Feature Generation?

**Методи валідації features**:

1. **t-SNE/UMAP візуалізація**: Чи формують features чіткі кластери класів? (наша основна метрика)
2. **Linear classifier**: Натренувати простий linear/logistic regression на features - якщо точність висока, features хороші
3. **Cluster metrics**: Silhouette score, Davies-Bouldin index для вимірювання якості кластеризації
4. **Dimensionality**: PCA explained variance - чи зберігають перші N компонент більшість інформації?
5. **Transfer learning**: Використати features для іншої задачі - чи допомагають вони?
6. **Feature visualization**: Візуалізувати top activations для кожного feature channel

**У нашому випадку**: t-SNE показує чіткі кластери для Siamese embeddings → features корисні.

### 10. Які параметри необхідно враховувати при виборі архітектури CNN для задачі класифікації?

**Ключові параметри**:

1. **Input size**: 28×28 (Fashion-MNIST) vs 224×224 (ImageNet) - впливає на кількість pooling layers
2. **Number of classes**: 10 (Fashion-MNIST) - визначає output layer size
3. **Model capacity**: Баланс між underfitting (мало параметрів) і overfitting (багато параметрів)
4. **Depth vs Width**: Глибші мережі (більше layers) vs ширші (більше filters) - trade-off між gradient flow і capacity
5. **Kernel sizes**: 3×3 стандарт, 1×1 для bottlenecks, 5×5/7×7 для початкових layers
6. **Pooling strategy**: MaxPooling (features preservation) vs AvgPooling (smoothing)
7. **Skip connections**: ResNet/DenseNet style для gradient flow
8. **Batch Normalization**: Майже завжди корисна для стабільності
9. **Computational budget**: Доступна GPU memory, training time, inference speed requirements

### 11. Як можна покращити точність класифікації за допомогою налаштування моделі?

**Практичні техніки** (застосовані в нашому lab):

1. **Hyperparameter tuning**:
   - Growth rate (k): 12 → 16 (більше capacity)
   - Dropout rate: 0.2 → 0.3 (більше regularization)
   - Learning rate: Grid search [0.001, 0.0005, 0.0001]
2. **Advanced augmentation**: MixUp, CutMix, AutoAugment для складніших варіацій
3. **Ensemble**: Тренувати кілька моделей з різною ініціалізацією/архітектурою, усереднювати predictions
4. **Label smoothing**: Замість hard labels (0/1) використати soft labels (0.1/0.9) - зменшує overconfidence
5. **Test-time augmentation**: Застосувати augmentation під час inference, усереднити predictions
6. **Learning rate warmup**: Поступово збільшувати LR перші кілька epochs
7. **Focal loss**: Якщо class imbalance - фокусуватись на важких прикладах
8. **Architecture search**: NAS (Neural Architecture Search) для автоматичного пошуку оптимальної архітектури

**Результат у нашому lab**: DenseNet з правильним тюнінгом досягає >90% accuracy на Fashion-MNIST.

---

**End of Explanations with Q&A**
