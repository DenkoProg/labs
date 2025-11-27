# Lab 4: YOLOv1 Object Detection - Deep Dive Explanations

## Table of Contents

1. [Overview](#overview)
2. [Object Detection Evolution](#object-detection-evolution)
   - [R-CNN](#r-cnn-region-based-cnn)
   - [Fast R-CNN](#fast-r-cnn)
   - [Faster R-CNN](#faster-r-cnn)
   - [YOLO: A Paradigm Shift](#yolo-a-paradigm-shift)
3. [YOLOv1 Architecture](#yolov1-architecture)
4. [YOLO Loss Function](#yolo-loss-function)
5. [Dataset Preparation](#dataset-preparation)
6. [Training Pipeline](#training-pipeline)
7. [Post-Processing: NMS](#post-processing-nms)
8. [Evaluation: mAP Metric](#evaluation-map-metric)
9. [Comparison with R-CNN Family](#comparison-with-r-cnn-family)
10. [Key Takeaways](#key-takeaways)
11. [FAQ](#frequently-asked-questions)

---

## Overview

This lab implements **YOLOv1 (You Only Look Once)** object detection from scratch, trained on the Open Images Dataset. Object detection is a fundamental computer vision task that involves:

1. **Localization**: Finding WHERE objects are (bounding boxes)
2. **Classification**: Determining WHAT objects are (class labels)

Unlike image classification which outputs a single label, object detection outputs multiple bounding boxes with class predictions.

**Lab Goals**:

- Implement YOLOv1 architecture (backbone + detection head)
- Create custom multi-task loss function (localization + confidence + classification)
- Train on Open Images Dataset using FiftyOne
- Implement Non-Maximum Suppression (NMS) for post-processing
- Evaluate with mean Average Precision (mAP) metric

---

## Object Detection Evolution

Before diving into YOLO, it's essential to understand the evolution of object detection methods that preceded it.

### R-CNN (Region-based CNN)

**Published**: 2014 by Ross Girshick et al.

**The Problem**: How do we find objects in an image when we don't know how many there are or where they are?

**R-CNN Solution**: A two-stage approach

```
Image → Region Proposals → CNN Features → Classification + Bounding Box Regression
```

#### Step 1: Region Proposal (Selective Search)

**Selective Search Algorithm**:

1. Over-segment image into superpixels using graph-based segmentation
2. Greedily merge similar regions based on:
   - Color similarity
   - Texture similarity
   - Size similarity
   - Shape compatibility
3. Output ~2000 region proposals per image

```python
# Conceptual pseudo-code
def selective_search(image):
    segments = segment_image(image)  # ~1000-2000 initial segments
    regions = []

    while len(segments) > 1:
        # Find most similar pair
        i, j = find_most_similar(segments)
        # Merge them
        merged = merge(segments[i], segments[j])
        regions.append(get_bbox(merged))
        segments.remove(i, j)
        segments.append(merged)

    return regions  # ~2000 proposals
```

#### Step 2: Feature Extraction (CNN)

For each of the ~2000 region proposals:

1. **Warp** region to fixed size (227×227 for AlexNet)
2. **Extract features** using pre-trained CNN (AlexNet)
3. Output: 4096-dimensional feature vector per region

```
Region (arbitrary size) → Warp → CNN → 4096-D feature vector
```

#### Step 3: Classification (SVM)

- Train one **binary SVM per class**
- Each SVM predicts: "Is this region class X or not?"
- Output: Class scores for each region

#### Step 4: Bounding Box Regression

- Learn to **refine** proposal coordinates
- Linear regression from CNN features to box offsets (dx, dy, dw, dh)
- Improves localization accuracy

**R-CNN Pipeline Summary**:

```
Image
  ↓
Selective Search → ~2000 proposals
  ↓
For each proposal:
  ├── Warp to 227×227
  ├── CNN forward pass (4096-D features)
  ├── SVM classification (C classes)
  └── Bounding box regression (4 offsets)
  ↓
Post-processing: NMS to remove duplicates
  ↓
Final detections
```

**R-CNN Problems**:

| Issue                      | Impact                                   |
| -------------------------- | ---------------------------------------- |
| **Slow training**          | SVM + bbox regressor trained separately  |
| **Slow inference**         | ~47 seconds per image (2000 CNN passes!) |
| **Storage hungry**         | Features for all proposals must be saved |
| **No end-to-end training** | Pipeline stages trained independently    |

---

### Fast R-CNN

**Published**: 2015 by Ross Girshick

**Key Innovation**: Share CNN computation across all proposals

```
Image → CNN (once) → Feature Map → RoI Pooling → FC layers → Classification + Regression
```

#### RoI Pooling (Region of Interest Pooling)

The breakthrough that enables sharing computation:

**Problem**: Each proposal has different size, but FC layers need fixed input

**Solution**: RoI Pooling converts any-size region to fixed-size output

```python
def roi_pooling(feature_map, roi, output_size=(7, 7)):
    """
    feature_map: H×W×C from CNN
    roi: (x1, y1, x2, y2) region coordinates
    output_size: fixed size for FC layers
    """
    # Extract region from feature map
    region = feature_map[y1:y2, x1:x2]

    # Divide into output_size grid
    h_step = (y2 - y1) / output_size[0]
    w_step = (x2 - x1) / output_size[1]

    # Max pool each grid cell
    output = np.zeros(output_size + (C,))
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            cell = region[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            output[i, j] = cell.max(axis=(0, 1))

    return output  # 7×7×C regardless of input size
```

#### Multi-task Loss

Fast R-CNN introduces **joint training** of classification and localization:

$$L = L_{cls} + \lambda \cdot L_{loc}$$

**Classification Loss** (Softmax Cross-Entropy):
$$L_{cls} = -\log(p_{true\_class})$$

**Localization Loss** (Smooth L1):
$$L_{loc} = \sum_{i \in \{x,y,w,h\}} smooth_{L1}(t_i - t_i^*)$$

Where smooth L1 is:
$$smooth_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

**Why Smooth L1?**

- Less sensitive to outliers than L2 (MSE)
- Doesn't explode for large errors
- Converges faster

**Fast R-CNN Improvements over R-CNN**:

| Metric         | R-CNN      | Fast R-CNN   |
| -------------- | ---------- | ------------ |
| Training time  | 84 hours   | 9.5 hours    |
| Inference time | 47 sec/img | 0.32 sec/img |
| mAP (VOC 2007) | 66.0%      | 66.9%        |

**Remaining Bottleneck**: Selective Search still takes ~2 seconds per image!

---

### Faster R-CNN

**Published**: 2015 by Shaoqing Ren et al.

**Key Innovation**: Replace Selective Search with a neural network — **Region Proposal Network (RPN)**

```
Image → Backbone CNN → Feature Map ─┬─→ RPN → Proposals
                                     └─→ RoI Pooling → Classification + Regression
```

#### Region Proposal Network (RPN)

A small network that slides over the feature map and predicts:

1. **Objectness score**: Is there an object here? (binary)
2. **Box coordinates**: Refined bounding box

**Anchor Boxes**: Pre-defined reference boxes at each spatial location

```python
# Anchor configuration
SCALES = [128, 256, 512]      # Different sizes
RATIOS = [0.5, 1.0, 2.0]      # Different aspect ratios
NUM_ANCHORS = 9               # 3 scales × 3 ratios

# At each feature map location, we have 9 anchors
# Feature map 50×38 → 50×38×9 = 17,100 proposals
```

**RPN Architecture**:

```
Feature Map (H×W×C)
      ↓
3×3 Conv (256 filters)  # Sliding window
      ↓
  ┌───┴───┐
  ↓       ↓
1×1 Conv  1×1 Conv
(2k)      (4k)
  ↓       ↓
Objectness Box coords
scores    (dx,dy,dw,dh)
```

**RPN Loss**:
$$L_{RPN} = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

Where:

- $p_i$ = predicted objectness probability for anchor $i$
- $p_i^*$ = ground truth (1 if IoU > 0.7, 0 if IoU < 0.3)
- $t_i$ = predicted box coordinates
- $t_i^*$ = ground truth box coordinates

#### Training Strategy

**4-Step Alternating Training**:

1. Train RPN (initialized from ImageNet pre-trained backbone)
2. Train Fast R-CNN using RPN proposals (separate backbone)
3. Fix shared conv layers, fine-tune RPN
4. Fix shared conv layers, fine-tune Fast R-CNN

**End-to-End Training** (later): Joint training with approximate backprop through RoI pooling

**Faster R-CNN Performance**:

| Metric         | Fast R-CNN       | Faster R-CNN    |
| -------------- | ---------------- | --------------- |
| Inference time | 2.3 sec          | 0.2 sec (5 fps) |
| mAP (VOC 2007) | 66.9%            | 69.9%           |
| Proposals      | Selective Search | RPN (learned)   |

**Still Two-Stage**: Proposals first, then classification → inherent latency

---

### YOLO: A Paradigm Shift

**Published**: 2015 by Joseph Redmon et al.

**Revolutionary Idea**: Treat object detection as a **single regression problem**

Instead of:

```
Image → Propose regions → Classify each region
```

YOLO does:

```
Image → Single CNN → All detections at once
```

**Key Insight**: "You Only Look Once" — one forward pass through the network predicts all bounding boxes and class probabilities simultaneously.

#### Why YOLO is Different

| Aspect               | R-CNN Family                   | YOLO                   |
| -------------------- | ------------------------------ | ---------------------- |
| **Approach**         | Two-stage (propose + classify) | Single-stage (unified) |
| **Speed**            | 0.5-7 fps                      | 45-155 fps             |
| **Global reasoning** | Each region independently      | Full image context     |
| **Training**         | Multi-stage pipeline           | End-to-end             |

#### YOLO Grid-based Detection

YOLO divides image into **S×S grid** (default S=7):

```
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ ★ │   │   │   │   │  ← Cell containing object center
├───┼───┼───┼───┼───┼───┼───┤      is responsible for detecting it
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
```

**Each grid cell predicts**:

- **B bounding boxes** (default B=2), each with:
  - $(x, y)$: Center coordinates relative to grid cell
  - $(w, h)$: Width and height relative to image
  - $confidence$: $P(Object) \times IoU_{pred}^{truth}$
- **C class probabilities**: $P(Class_i | Object)$

**Output tensor shape**: $S \times S \times (B \times 5 + C)$

For our implementation: $7 \times 7 \times (2 \times 5 + 3) = 7 \times 7 \times 13$

---

## YOLOv1 Architecture

### Original Architecture (Darknet)

The original YOLOv1 uses a custom architecture inspired by GoogLeNet:

```
Input: 448×448×3

Layer 1:  Conv 7×7×64 stride=2, MaxPool 2×2      → 112×112×64
Layer 2:  Conv 3×3×192, MaxPool 2×2              → 56×56×192
Layer 3:  Conv 1×1×128, Conv 3×3×256, Conv 1×1×256, Conv 3×3×512, MaxPool → 28×28×512
Layer 4:  [Conv 1×1×256, Conv 3×3×512]×4, Conv 1×1×512, Conv 3×3×1024, MaxPool → 14×14×1024
Layer 5:  [Conv 1×1×512, Conv 3×3×1024]×2, Conv 3×3×1024, Conv 3×3×1024 stride=2 → 7×7×1024
Layer 6:  Conv 3×3×1024, Conv 3×3×1024           → 7×7×1024
Layer 7:  Flatten → FC 4096 → FC S×S×(B×5+C)     → 7×7×30 (for VOC)

Total: 24 convolutional layers + 2 fully connected layers
```

### Our Implementation (ResNet34 Backbone)

We use transfer learning with pre-trained ResNet34:

```python
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=10, use_pretrained=True):
        if use_pretrained:
            # ResNet34 backbone (pre-trained on ImageNet)
            resnet = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])

            # Adaptive pooling to get S×S spatial dimensions
            self.adaptive_pool = nn.AdaptiveAvgPool2d((S, S))

            # Detection head
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * S * S, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, S * S * (B * 5 + C)),
            )
```

**Why ResNet34?**

1. **Pre-trained weights**: ImageNet features transfer well to detection
2. **Skip connections**: Better gradient flow, deeper network works better
3. **Proven architecture**: Widely used, well-understood
4. **Faster training**: Converges faster than training from scratch

**Architecture Flow**:

```
Input: 448×448×3
    ↓
ResNet34 Backbone (conv1 → layer4)
    ↓
512×14×14 feature map
    ↓
AdaptiveAvgPool2d(7, 7)
    ↓
512×7×7 = 25,088 features
    ↓
Flatten → FC(25088, 4096) → Dropout → LeakyReLU
    ↓
FC(4096, 637) → Reshape to 7×7×13
    ↓
Output: S×S×(B×5+C) = 7×7×13
```

### CNNBlock (For Custom Backbone)

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
```

**Why LeakyReLU(0.1)?**

- YOLO paper uses LeakyReLU throughout
- Prevents "dying ReLU" problem (gradients = 0 for negative inputs)
- $f(x) = \max(0.1x, x)$ allows small gradients for negative values

**Why BatchNorm?**

- Stabilizes training (normalizes activations)
- Allows higher learning rates
- Acts as regularization
- Modern addition (not in original YOLO paper)

---

## YOLO Loss Function

The YOLO loss is the heart of the training. It combines three components:

$$L = \lambda_{coord} \cdot L_{coord} + L_{conf} + L_{class}$$

### Visual Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOLO Loss Function                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─── Localization Loss (λ_coord = 5) ───┐                                  │
│  │                                        │                                  │
│  │  λ_coord × Σ Σ 𝟙_ij^obj [(x-x̂)² + (y-ŷ)²]                               │
│  │            i j                         │                                  │
│  │                                        │                                  │
│  │  + λ_coord × Σ Σ 𝟙_ij^obj [(√w-√ŵ)² + (√h-√ĥ)²]                         │
│  │              i j                       │                                  │
│  └────────────────────────────────────────┘                                  │
│                                                                              │
│  ┌─── Confidence Loss ───────────────────┐                                  │
│  │                                        │                                  │
│  │  Σ Σ 𝟙_ij^obj (C - Ĉ)²                │  ← Object present                │
│  │  i j                                   │                                  │
│  │                                        │                                  │
│  │  + λ_noobj × Σ Σ 𝟙_ij^noobj (C - Ĉ)²  │  ← No object (λ_noobj = 0.5)    │
│  │              i j                       │                                  │
│  └────────────────────────────────────────┘                                  │
│                                                                              │
│  ┌─── Classification Loss ───────────────┐                                  │
│  │                                        │                                  │
│  │  Σ 𝟙_i^obj Σ (p(c) - p̂(c))²          │                                  │
│  │  i       c∈classes                     │                                  │
│  └────────────────────────────────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Loss Components Explained

#### 1. Localization Loss (Box Coordinates)

```python
# Coordinates (x, y) - center of box relative to cell
box_loss_xy = self.mse(pred_xy, target_xy)  # (x - x̂)² + (y - ŷ)²

# Dimensions (w, h) - use sqrt to reduce impact of large boxes
pred_wh = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)
target_wh = torch.sqrt(target_wh)
box_loss_wh = self.mse(pred_wh, target_wh)  # (√w - √ŵ)² + (√h - √ĥ)²
```

**Why sqrt for width/height?**

Consider two scenarios:

- Large box: 100×100, error of 10 pixels → 10% error
- Small box: 10×10, error of 10 pixels → 100% error!

Without sqrt, the same absolute error has equal contribution. With sqrt:

- Large: $(\sqrt{100} - \sqrt{90})^2 = (10 - 9.49)^2 = 0.26$
- Small: $(\sqrt{10} - \sqrt{0})^2 = (3.16 - 0)^2 = 10$

**$\lambda_{coord} = 5$**: Weight localization more heavily (most important task)

#### 2. Confidence Loss

```python
# For cells WITH objects
object_loss = self.mse(
    exists_box * pred_confidence,
    exists_box * target_confidence  # Target = 1 × IoU
)

# For cells WITHOUT objects (most cells!)
no_object_loss = self.mse(
    (1 - exists_box) * pred_confidence,
    (1 - exists_box) * target_confidence  # Target = 0
)
```

**Why $\lambda_{noobj} = 0.5$?**

In a 7×7 grid, most cells don't contain objects:

- With 3 objects: 3 positive cells, 46 negative cells
- Without weighting, negative gradient would dominate
- $\lambda_{noobj} = 0.5$ balances positive/negative contributions

**Confidence target**: $C^* = P(Object) \times IoU_{pred}^{truth}$

- If no object: $C^* = 0$
- If object: $C^* = IoU$ (model learns to predict its own accuracy!)

#### 3. Classification Loss

```python
class_loss = self.mse(
    exists_box * pred_class_probs,
    exists_box * target_class_probs  # One-hot encoded
)
```

**Conditional probability**: $P(Class_i | Object)$

- Only computed for cells containing objects
- Target is one-hot vector: [0, 0, 1, 0, ...] for class 2

### Responsible Box Selection

Each cell predicts B boxes, but only ONE is responsible for each object:

```python
# Compute IoU for both predicted boxes
iou_b1 = intersection_over_union(predictions[..., :4], targets[..., :4])
iou_b2 = intersection_over_union(predictions[..., 5:9], targets[..., :4])

# Select box with higher IoU
ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
_, best_box = torch.max(ious, dim=0)  # 0 or 1

# Use selected box for loss computation
box_predictions = best_box * predictions[..., 5:9] + (1 - best_box) * predictions[..., :4]
```

**Why?** Encourages specialization:

- Box 1 might specialize in tall objects
- Box 2 might specialize in wide objects
- During training, each box learns different aspect ratios

### IoU (Intersection over Union)

```python
def intersection_over_union(boxes_pred, boxes_target):
    # Convert from (x_center, y_center, w, h) to (x1, y1, x2, y2)
    box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
    box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
    box1_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
    box1_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2
    # ... same for box2 ...

    # Intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Areas
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)
```

**IoU Interpretation**:

- IoU = 1.0: Perfect overlap
- IoU = 0.5: Acceptable detection (common threshold)
- IoU = 0.0: No overlap

```
┌─────────────────────┐
│    Ground Truth     │
│   ┌─────────────┐   │
│   │ Intersection│   │
│   │             │   │
│   └─────────────┘   │
│        Prediction   │
└─────────────────────┘

IoU = Intersection / Union = Intersection / (A + B - Intersection)
```

---

## Dataset Preparation

### Open Images Dataset with FiftyOne

We use FiftyOne library to download and manage Open Images V7:

```python
def setup_open_images_dataset(classes=None, max_samples=2000, split="train"):
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=classes,  # ["Dog", "Cat", "Bird"]
        max_samples=max_samples,
        shuffle=True,
    )
    return dataset
```

**Why FiftyOne?**

1. **Easy access**: Downloads specific classes on-demand
2. **Standardized format**: Consistent bounding box format
3. **Visualization**: Built-in tools for debugging
4. **Filtering**: Easy to filter by class, confidence, etc.

### YOLO Target Encoding

Converting bounding boxes to YOLO format:

```python
def _convert_to_yolo_target(self, boxes, labels):
    """
    boxes: List of [x_center, y_center, width, height] in [0, 1]
    labels: List of class indices
    """
    target = torch.zeros(S, S, B * 5 + C)

    for box, label in zip(boxes, labels):
        x, y, w, h = box

        # Find responsible cell
        cell_x = int(x * S)  # Column index (0-6)
        cell_y = int(y * S)  # Row index (0-6)

        # Convert to cell-relative coordinates
        x_cell = x * S - cell_x  # Offset within cell [0, 1]
        y_cell = y * S - cell_y
        w_cell = w * S  # Width relative to grid (can be > 1)
        h_cell = h * S

        # Only assign if cell is empty (one object per cell limitation)
        if target[cell_y, cell_x, 4] == 0:
            target[cell_y, cell_x, :4] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
            target[cell_y, cell_x, 4] = 1  # Confidence = 1 (object present)
            target[cell_y, cell_x, 10 + label] = 1  # One-hot class

    return target
```

**Coordinate System**:

```
Image coordinates [0, 1]:
┌───────────────────────────────┐
│(0,0)                    (1,0) │
│                               │
│         ● (0.5, 0.5)         │
│                               │
│(0,1)                    (1,1) │
└───────────────────────────────┘

Grid cell assignment (S=7):
cell_x = floor(x * 7)
cell_y = floor(y * 7)

Cell-relative coordinates:
x_cell = x * 7 - cell_x  ∈ [0, 1)
y_cell = y * 7 - cell_y  ∈ [0, 1)
```

### Data Augmentation Considerations

**Important**: Standard image augmentations can break object detection!

```python
# PROBLEMATIC: RandomAffine transforms image but not boxes!
transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2))
```

**Why it's problematic**:

- Image shifts 20% right
- Bounding boxes still reference original positions
- Network learns incorrect labels → poor performance

**Solution**: Use bbox-aware augmentation (albumentations) or disable geometric transforms:

```python
# Safe augmentations (pixel-level only)
transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Training Pipeline

### Optimizer: SGD with Momentum

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    weight_decay=5e-4
)
```

**Why SGD for Detection?**

- More stable than Adam for detection tasks
- Momentum (0.9) helps navigate loss landscape
- Weight decay (5e-4) prevents overfitting

### Learning Rate Schedule

Following the YOLO paper's schedule:

```python
LR_SCHEDULE = [
    (10, 1e-3, 1e-2),   # Warmup: slowly increase LR
    (75, 1e-2, 1e-2),   # Main training: constant high LR
    (30, 1e-3, 1e-3),   # Fine-tuning: reduced LR
    (30, 1e-4, 1e-4),   # Final polish: very low LR
]
```

**Warmup Phase**:

- Start with low LR (1e-3)
- Gradually increase to target (1e-2)
- Prevents early divergence with high LR

```
LR
 │
1e-2│         ┌────────────────┐
    │        /                  \
1e-3│       /                    └───────┐
    │      /                              └───────
1e-4│     /
    └─────┼─────┼─────────────────┼───────┼───────→ Epochs
          10    85               115     145
```

### Training Loop

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in tqdm(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
```

**Gradient Clipping**: Prevents exploding gradients

- YOLO loss can produce large gradients (especially early in training)
- Clip norm to 1.0 keeps updates bounded

---

## Post-Processing: NMS

### The Problem

YOLO outputs $S \times S \times B = 7 \times 7 \times 2 = 98$ bounding boxes per image. Many overlap!

```
Before NMS:
┌─────────────────────────────┐
│    ┌───────┐                │
│    │ ┌─────┴─┐              │
│    │ │ ┌─────┴─┐            │
│    │ │ │  DOG  │            │
│    │ │ └───────┘            │
│    │ └─────────┘            │
│    └───────────┘            │
│                             │
└─────────────────────────────┘
Multiple overlapping boxes for same object
```

### Non-Maximum Suppression Algorithm

```python
def non_max_suppression(boxes, iou_threshold=0.5, conf_threshold=0.4):
    # Step 1: Filter by confidence
    boxes = [box for box in boxes if box[1] > conf_threshold]

    # Step 2: Sort by confidence (descending)
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)

    boxes_after_nms = []

    while boxes:
        # Step 3: Take highest confidence box
        chosen_box = boxes.pop(0)
        boxes_after_nms.append(chosen_box)

        # Step 4: Remove boxes with high IoU (same class only)
        boxes = [
            box for box in boxes
            if box[0] != chosen_box[0]  # Different class: keep
            or calculate_iou(chosen_box[2:], box[2:]) < iou_threshold  # Low IoU: keep
        ]

    return boxes_after_nms
```

**NMS Walkthrough**:

```
Step 1: Filter conf > 0.4
┌─────────────────────────────────────────────────────────┐
│ Box A: Dog 0.9  Box B: Dog 0.7  Box C: Dog 0.3 (removed)│
│ Box D: Cat 0.8                                          │
└─────────────────────────────────────────────────────────┘

Step 2: Sort by confidence
[A(0.9), D(0.8), B(0.7)]

Step 3-4: Iterate
- Take A (Dog 0.9) → keep
- Check D (Cat 0.8): different class → keep
- Check B (Dog 0.7): IoU(A, B) = 0.85 > 0.5 → remove

Result: [A(Dog 0.9), D(Cat 0.8)]
```

### Cells to Boxes Conversion

```python
def cells_to_boxes(predictions, S=7, B=2, C=10):
    """Convert YOLO grid output to list of bounding boxes."""
    all_boxes = []

    for batch_idx in range(batch_size):
        boxes = []

        for i in range(S):  # Rows
            for j in range(S):  # Columns
                # Get class probabilities
                class_probs = predictions[batch_idx, i, j, 10:]
                best_class = torch.argmax(class_probs).item()
                best_prob = class_probs[best_class].item()

                for b in range(B):  # Each bounding box
                    box_offset = b * 5

                    # Convert cell-relative to image coordinates
                    x = (predictions[..., box_offset + 0] + j) / S
                    y = (predictions[..., box_offset + 1] + i) / S
                    w = predictions[..., box_offset + 2] / S
                    h = predictions[..., box_offset + 3] / S
                    confidence = predictions[..., box_offset + 4].item()

                    # Final score = confidence × class probability
                    score = confidence * best_prob

                    boxes.append([best_class, score, x, y, w, h])

        all_boxes.append(boxes)

    return all_boxes
```

---

## Evaluation: mAP Metric

### What is mAP?

**mean Average Precision (mAP)**: The standard metric for object detection

```
mAP = (1/C) × Σ AP_c
         c∈classes
```

Where AP (Average Precision) is the area under the Precision-Recall curve.

### Computing AP for One Class

```python
def calculate_ap(detections, ground_truths, iou_threshold=0.5):
    # Sort detections by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    TP = np.zeros(len(detections))
    FP = np.zeros(len(detections))

    for det_idx, detection in enumerate(detections):
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt['matched']:
                continue

            iou = calculate_iou(detection['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            TP[det_idx] = 1
            ground_truths[best_gt_idx]['matched'] = True
        else:
            FP[det_idx] = 1

    # Compute precision and recall at each threshold
    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(FP)

    precision = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
    recall = TP_cumsum / (len(ground_truths) + 1e-6)

    # Compute AP using trapezoidal rule
    ap = np.trapz(precision, recall)

    return ap
```

### Precision-Recall Curve

```
Precision
    │
1.0 ├────────┐
    │        │
0.8 ├        └───┐
    │            │
0.6 ├            └───────┐
    │                    │
0.4 ├                    └───────┐
    │                            │
0.2 ├                            └───────
    │
0.0 └────────┼────────┼────────┼────────┼──→ Recall
             0.25     0.5      0.75     1.0

AP = Area under this curve
```

**Interpretation**:

- High AP: Good precision maintained as recall increases
- Perfect detector: Precision = 1.0 for all recall values
- Random detector: AP ≈ 0.1 (assuming 10% positive rate)

### mAP@0.5 vs mAP@0.5:0.95

- **mAP@0.5**: IoU threshold = 0.5 (PASCAL VOC standard)
- **mAP@0.5:0.95**: Average over IoU thresholds [0.5, 0.55, ..., 0.95] (COCO standard)

COCO mAP is stricter — requires tighter bounding boxes.

---

## Comparison with R-CNN Family

### Speed vs Accuracy Trade-off

| Model         | Inference Speed        | mAP (VOC 2007) | Real-time? |
| ------------- | ---------------------- | -------------- | ---------- |
| R-CNN         | 47 sec                 | 66.0%          | ❌         |
| Fast R-CNN    | 2.3 sec                | 66.9%          | ❌         |
| Faster R-CNN  | 0.2 sec (5 fps)        | 69.9%          | ❌         |
| **YOLOv1**    | **0.022 sec (45 fps)** | 63.4%          | ✅         |
| YOLOv1 (Fast) | 0.007 sec (155 fps)    | 52.7%          | ✅         |

### Strengths of YOLO

1. **Speed**: ~1000× faster than R-CNN
2. **Global reasoning**: Sees entire image, fewer background errors
3. **Generalizable**: Better transfer to new domains (art, etc.)
4. **Simple**: Single network, end-to-end training

### Weaknesses of YOLO

1. **Small objects**: 7×7 grid can't handle many small objects
2. **Unusual aspect ratios**: Limited box shapes
3. **Localization accuracy**: Coarse grid reduces precision
4. **One object per cell**: Struggles with crowds

### When to Use What

| Scenario               | Best Choice                        |
| ---------------------- | ---------------------------------- |
| Real-time video        | YOLO                               |
| Autonomous driving     | YOLO (speed critical)              |
| Medical imaging        | Faster R-CNN (accuracy critical)   |
| Small object detection | Faster R-CNN                       |
| Edge devices           | YOLO variants (MobileNet backbone) |

---

## Key Takeaways

### Object Detection Evolution

✓ **R-CNN**: Introduced region-based detection, but very slow
✓ **Fast R-CNN**: Shared CNN computation, much faster
✓ **Faster R-CNN**: Replaced Selective Search with RPN, near real-time
✓ **YOLO**: Unified detection in single pass, true real-time

### YOLOv1 Design Principles

✓ **Grid-based**: Divide image into S×S cells
✓ **Multi-task**: Single loss combines localization, confidence, classification
✓ **Anchors (implicit)**: B boxes per cell with different specializations
✓ **End-to-end**: Directly optimizes detection objective

### Implementation Details

✓ **Backbone**: Transfer learning from ImageNet (ResNet34)
✓ **Loss weighting**: λ_coord=5, λ_noobj=0.5 balance loss components
✓ **sqrt(w,h)**: Normalize scale sensitivity for different box sizes
✓ **NMS**: Post-processing to remove duplicate detections

### Training Best Practices

✓ **Learning rate warmup**: Prevents early divergence
✓ **Gradient clipping**: Stabilizes training
✓ **bbox-aware augmentation**: Must transform boxes with images
✓ **Evaluation during training**: Monitor mAP, not just loss

---

## Frequently Asked Questions

### 1. Чому YOLO швидший за R-CNN?

**YOLO швидший через однопрохідну архітектуру**:

- **R-CNN**: ~2000 region proposals × CNN forward pass = ~2000 CNN passes
- **YOLO**: 1 forward pass → всі детекції одразу

YOLO розглядає детекцію як **регресійну задачу**: один прохід через мережу видає всі bounding boxes і класи. Це усуває:

1. Окремий етап генерації proposals
2. Багаторазові прогони CNN
3. Окреме тренування компонентів

**Порівняння швидкості**:

- R-CNN: 47 секунд/зображення
- YOLO: 0.022 секунди/зображення (~2000× швидше!)

### 2. Як YOLO визначає, яка комірка "відповідальна" за об'єкт?

**Комірка, що містить центр об'єкта, є відповідальною**:

```python
cell_x = int(x_center * S)  # Індекс колонки (0-6)
cell_y = int(y_center * S)  # Індекс рядка (0-6)
```

Якщо центр bounding box знаходиться в комірці (3, 4), тільки ця комірка передбачатиме цей об'єкт. Інші комірки отримують target=0 для confidence.

**Проблема**: Якщо два об'єкти мають центри в одній комірці, YOLO може передбачити тільки один (обмеження архітектури).

### 3. Чому в loss функції використовується sqrt для ширини/висоти?

**Sqrt нормалізує вплив помилок для різних розмірів**:

Без sqrt:

- Великий box 100×100, помилка 10px: MSE = 100
- Малий box 10×10, помилка 10px: MSE = 100 (та сама!)

Але відносна помилка різна: 10% vs 100%!

З sqrt:

- Великий: (√100 - √90)² = 0.26
- Малий: (√10 - √0)² = 10

Тепер модель більше штрафується за помилки на малих об'єктах, що інтуїтивно правильно.

### 4. Що таке NMS і навіщо він потрібен?

**Non-Maximum Suppression видаляє дублікати детекцій**:

YOLO передбачає 98 boxes (7×7×2). Багато з них перекриваються для одного об'єкта. NMS:

1. Сортує boxes за confidence (від найвищого)
2. Бере найкращий box
3. Видаляє всі boxes того ж класу з IoU > threshold
4. Повторює поки є boxes

**Приклад**:

- Dog boxes: A(0.9), B(0.7), C(0.5)
- IoU(A,B) = 0.8 > 0.5 → видаляємо B
- IoU(A,C) = 0.3 < 0.5 → залишаємо C
- Результат: A, C

### 5. Яка різниця між λ_coord і λ_noobj?

**λ_coord = 5**: Збільшує важливість локалізації

Точне положення box важливіше за confidence. Без цього множника, loss для координат був би занизьким порівняно з classification loss.

**λ_noobj = 0.5**: Зменшує вплив "пустих" комірок

В 7×7 grid більшість комірок не містять об'єктів:

- 3 об'єкти = 3 positive cells, 46 negative cells
- Без λ_noobj негативний градієнт домінував би
- λ_noobj=0.5 балансує внески

### 6. Як працює IoU і чому це важлива метрика?

**IoU (Intersection over Union)** вимірює перекриття двох boxes:

$$IoU = \frac{Area_{intersection}}{Area_{union}}$$

**Інтерпретація**:

- IoU = 1.0: Ідеальне перекриття
- IoU = 0.5: Прийнятна детекція (стандартний поріг)
- IoU = 0.0: Немає перекриття

**Використання в YOLO**:

1. **Вибір відповідального box**: Box з вищим IoU тренується
2. **NMS**: Видаляє boxes з IoU > threshold
3. **mAP**: Detection вважається TP якщо IoU ≥ threshold

### 7. Чому mAP використовується замість accuracy для object detection?

**Accuracy не підходить для detection** через:

1. **Variable number of objects**: Кількість об'єктів різна на кожному зображенні
2. **Localization matters**: Не достатньо знати клас, треба точне положення
3. **Confidence ranking**: Хочемо знати, наскільки модель впевнена

**mAP враховує**:

- Precision: Скільки детекцій правильні?
- Recall: Скільки об'єктів знайдено?
- Ranking: Чи впевненіші predictions точніші?

### 8. Як transfer learning допомагає в object detection?

**Transfer learning прискорює тренування і покращує результати**:

ImageNet pre-trained backbone вже вміє:

- Виявляти edges, textures, shapes
- Розрізняти об'єкти від фону
- Ієрархічно представляти features

**Без transfer learning**:

- Треба тренувати з нуля
- Потрібно більше даних
- Довше converge

**З transfer learning**:

- Backbone заморожений або fine-tuned з низьким LR
- Тільки detection head тренується "з нуля"
- Швидка конвергенція, кращі результати

### 9. Які обмеження YOLOv1?

**Основні обмеження**:

1. **Один об'єкт на комірку**: Якщо 2 об'єкти в одній комірці, передбачить тільки 1
2. **Грубий grid**: 7×7 = 49 комірок, погано для маленьких об'єктів
3. **Aspect ratios**: Тільки 2 boxes на комірку, обмежені форми
4. **Localization accuracy**: Coarse predictions порівняно з Faster R-CNN

**Вирішено в наступних версіях**:

- YOLOv2: Anchor boxes, batch normalization, multi-scale
- YOLOv3: Feature Pyramid Network, 3 scales
- YOLOv4+: CSPDarknet, PANet, advanced augmentation

### 10. Як правильно аугментувати дані для object detection?

**Проблема**: Стандартні аугментації зміщують зображення, але не boxes!

```python
# НЕПРАВИЛЬНО: boxes не трансформуються
transforms.RandomAffine(translate=(0.2, 0.2))
```

**Правильний підхід** — bbox-aware augmentation (albumentations):

```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=400, height=400),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo'))

# Boxes автоматично трансформуються разом із зображенням
transformed = transform(image=image, bboxes=bboxes)
```

**Безпечні аугментації** (не потребують bbox transform):

- ColorJitter (brightness, contrast, saturation)
- Normalize
- GaussianBlur

---

## Further Reading

### Papers

1. **YOLO**: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (Redmon et al., 2015)
2. **R-CNN**: [Rich feature hierarchies for accurate object detection](https://arxiv.org/abs/1311.2524) (Girshick et al., 2014)
3. **Fast R-CNN**: [Fast R-CNN](https://arxiv.org/abs/1504.08083) (Girshick, 2015)
4. **Faster R-CNN**: [Faster R-CNN: Towards Real-Time Object Detection](https://arxiv.org/abs/1506.01497) (Ren et al., 2015)
5. **YOLOv2**: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) (Redmon & Farhadi, 2016)
6. **YOLOv3**: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) (Redmon & Farhadi, 2018)

### Concepts

- **Anchor Boxes**: Pre-defined reference shapes for detection
- **Feature Pyramid Networks**: Multi-scale feature extraction
- **Focal Loss**: Handling class imbalance in detection
- **IoU Loss**: Alternative to MSE for bounding box regression

### Extensions

1. **YOLOv4/v5/v7/v8**: Modern YOLO variants with SOTA performance
2. **SSD**: Single Shot MultiBox Detector (similar to YOLO)
3. **RetinaNet**: Focal loss for one-stage detection
4. **DETR**: Transformer-based detection
5. **CenterNet**: Anchor-free detection

---

**End of Explanations**
