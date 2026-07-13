# DocuSort AI
Automated scanned document classification using a custom-trained CNN and a fine-tuned ResNet18, with Grad-CAM interpretability and confidence-based routing for uncertain predictions.

## Overview
DocuSort AI classifies scanned document images into 10 categories (Advertisement, Email, Form, Letter, Memo, News, Note, Report, Resume, Scientific) using two independently trained deep learning models, allowing direct comparison between a from-scratch architecture and transfer learning.
The project goes beyond training a single model — it includes comparative error analysis, model interpretability via Grad-CAM, and a production-oriented confidence-routing system that flags uncertain predictions for human review instead of forcing a guess.

## Problem Statement
Organizations that process large volumes of scanned documents (legal firms, banks, insurance companies) often need automated document routing/sorting. Manual sorting is slow and error-prone; DocuSort AI demonstrates an automated approach, along with the interpretability and safety mechanisms needed for real-world deployment.

## Dataset
Tobacco3482 — a subset of the Legacy Tobacco Document Library, containing 3,482 scanned document images across 10 classes. Chosen specifically because it is a genuine, real-world scanned document corpus (rather than a synthetic or overused benchmark dataset), and because its natural class imbalance (ranging from 620 images for "Memo" down to 120 for "Resume") provided a realistic opportunity to handle imbalanced classification properly.
Data split: 70% train / 15% validation / 15% test, using stratified sampling to preserve class proportions across all three sets.

## Architecture & Approach

### 1. Custom CNN (built from scratch)
- 4 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool), with channel depth increasing 32 → 64 → 128 → 256
- Global Average Pooling instead of a large flattened FC layer, to reduce overfitting risk on a small dataset
- Dropout (p=0.5) before the final classification layer
- ~391K parameters
- Full manual training loop written in PyTorch (forward pass, loss computation, backpropagation, optimizer step) — no high-level `.fit()` abstraction

### 2. ResNet18 (Transfer Learning)
- Pretrained ImageNet weights, first convolutional layer adapted for single-channel grayscale input (by averaging pretrained RGB filter weights)
- Backbone frozen; only the final classification layer fine-tuned on Tobacco3482
- ~11.2M total parameters, ~5,130 trainable

### 3. Training Details (both models)
- Weighted cross-entropy loss to address class imbalance
- Adam optimizer with `ReduceLROnPlateau` learning rate scheduling
- Early stopping based on validation macro-F1 (chosen over accuracy specifically because it treats all classes equally, exposing weaknesses on minority classes that accuracy alone would hide)

## Results
|           Model              |   Test Accuracy   |   Macro-F1   |
| ---------------------------- | ----------------- | ------------ |
| Custom CNN (from Scratch)    |       67.5%       |     0.684    |
| ResNet18 (Transfer Learning) |       70.6%       |     0.706    |


### Key Finding: Transfer Learning Did Not Uniformly Win
While ResNet18 achieved higher overall accuracy and macro-F1, per-class analysis revealed it underperformed the custom CNN specifically on the "Letter" class (recall dropped from 0.765 to 0.471). This suggests document-specific features learned from scratch may capture certain structural cues (e.g., salutation or signature placement) that generic ImageNet-pretrained features do not — overall metrics alone would have hidden this tradeoff.
Both models shared a persistent confusion cluster among Letter, Memo, and Report — classes with visually similar dense-paragraph layouts — indicating this is a genuine structural challenge in the dataset rather than an architecture-specific weakness.

## Model Interpretability (Grad-CAM)
Grad-CAM was implemented from scratch (using forward/backward hooks on the final convolutional layer) to visualize which regions of a document most influenced each model's prediction.
Notable finding: a "Letter" document misclassified as "Report" by both models was analyzed via Grad-CAM. The heatmaps revealed the models focused on numbered-list/tabular body content rather than letter-defining cues (salutation, signature). Further inspection showed this document itself blends letter and report characteristics — suggesting some errors reflect genuine label ambiguity in the source dataset, not pure model failure.

## Confidence-Based Routing
Rather than always forcing a prediction, the system flags low-confidence outputs for manual review — a pattern used in real production ML systems where misrouting has real costs.

| Threshold | % Flagged | Error Catch Rate | Errors Missed |
| --------- | --------- | ---------------- | ------------- |
| 0.45      |     38.4% |            65.3% |            59 |
| 0.50      |     47.6% |            76.5% |            40 |
| 0.60      |     62.3% |            91.8% |            14 |

A threshold of 0.6 was selected as the default, prioritizing catching real errors (91.8% catch rate) over minimizing manual review workload, on the reasoning that document misrouting carries higher real-world cost than manual review effort.

## Application
An interactive Streamlit app allows users to:
- Upload a scanned document image
- Toggle between the Custom CNN and ResNet18 models
- View prediction confidence and top-3 class probabilities
- See a Grad-CAM attention overlay showing model focus
- Adjust the confidence-routing threshold in real time
- Download a full PDF report of the classification result

## How to Run Locally
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/docusort-ai.git
cd docusort-ai
```

2. Set up a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

3. Download the [Tobacco3482 dataset](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) and place it under `data/raw/`.

4. Run the notebooks in `notebooks/` to train the models, or use pre-trained checkpoints if available.

5. Launch the app:
```bash
cd app
streamlit run streamlit_app.py
```

## Tech Stack
Python, PyTorch, TorchVision, Scikit-learn, Streamlit, OpenCV, Pillow, Matplotlib, Seaborn, FPDF2

