# Lab 3: DenseNet Classification + Siamese Networks for Image Search

**Goal**: Build DenseNet CNN for Fashion-MNIST classification (>90% accuracy), then use Siamese networks for similar image search with t-SNE visualization.

---

## Part 1: DenseNet Classification

### Data & Model

- [ ] Load Fashion-MNIST (10 classes, 28x28 grayscale)
- [ ] Normalize data, create train/val/test splits
- [ ] Implement DenseNet architecture:
  - Dense blocks with growth rate
  - Transition layers between blocks
  - Global average pooling + FC layer (10 classes)
- [ ] Add data augmentation (rotation, shift, zoom)

### Training & Evaluation

- [ ] Train with CrossEntropyLoss, Adam optimizer, LR scheduler
- [ ] Use dropout, weight decay for regularization
- [ ] Tune hyperparameters: growth rate, blocks, layers, learning rate, batch size
- [ ] Evaluate: accuracy, confusion matrix, precision/recall/F1
- [ ] Save best model weights

**Target**: >90% test accuracy

---

## Part 2: Siamese Network for Similar Images

### Architecture & Training

- [ ] Build Siamese network using DenseNet backbone
- [ ] Remove classification layer, add embedding layer (128-256 dims)
- [ ] Create pairs/triplets dataset (positive: same class, negative: different class)
- [ ] Implement Contrastive Loss or Triplet Loss
- [ ] Train embedding network, monitor loss and distances
- [ ] Save trained model

### Image Search

- [ ] Extract embeddings for all Fashion-MNIST images
- [ ] Implement search function: query → top-k similar images
- [ ] Use Euclidean distance or cosine similarity
- [ ] Test with sample queries
- [ ] Evaluate: top-k accuracy, mAP, visual inspection

---

## Part 3: t-SNE Visualization

- [ ] Generate embeddings for dataset (sample 5k-10k if needed)
- [ ] Apply t-SNE: perplexity=30-50, n_iter=1000+
- [ ] Create 2D scatter plot colored by class
- [ ] Analyze cluster quality and separations
- [ ] Compare visualizations:
  - Raw pixels (baseline)
  - DenseNet features
  - Siamese embeddings
- [ ] Show sample search results

---

## Implementation

### Environment

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### Files

- [ ] `lab3.ipynb` - Main implementation notebook
- [ ] `requirements.txt` - Dependencies

### Deliverables

1. Trained DenseNet model (>90% accuracy)
2. Trained Siamese network
3. Similar image search system
4. t-SNE visualizations
5. Results analysis

---

## Key Tips

**DenseNet**: Start small, use batch norm, monitor gradients
**Siamese**: Try margin values 0.2-1.0, validate embeddings during training
**t-SNE**: Use PCA pre-reduction for speed, try different perplexity values
**Performance**: Use GPU, efficient data loading, save checkpoints

---

## Timeline

- Part 1: 4-6 hours
- Part 2: 4-6 hours
- Part 3: 2-3 hours
- Documentation: 1-2 hours
- **Total**: 12-17 hours
