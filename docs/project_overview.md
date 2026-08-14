# Brain Tumor MRI Classification — Project Overview

## 1. Project Objective
This research project successfully designed, implemented, and optimized an end-to-end deep learning pipeline for multi-class brain magnetic resonance imaging (MRI) classification.

## 2. Problem Definition
Not explicitly documented in the notebook.

## 3. Dataset
The dataset utilized is the "Brain Tumor MRI Dataset" loaded from `/content/drive/MyDrive/brain_tumor_mri/Brain_Tumor_MRI_Dataset.zip`. It contains images divided into 'Train' and 'Test' folders.

## 4. Target Classes
- glioma
- meningioma
- pituitary
- notumor

## 5. Data Preparation
- The ZIP file is extracted to a local directory.
- The directory structure is verified to contain 8 folders (Train and Test splits for the 4 classes).
- A DataFrame is created storing the file paths, text labels, and split designation for each image.
- A numerical `label_id` is mapped to each class.

## 6. Exploratory Data Analysis
- Sample MRI images per class are plotted for both train and test splits.
- Class distribution is visualized using bar charts to count the number of images per class.
- An "Average MRI Appearance" (Mean Image Mosaic) is generated for each class by resizing sample images to 224x224 and averaging their pixel values.
- Pixel intensity analysis is conducted to plot the mean pixel brightness distribution (box plot) and intensity histogram per class. The channels are confirmed to be nearly identical (grayscale-like), but RGB is retained as ImageNet weights require 3-channel input.

## 7. Data Splitting
- The 'Train' split from the source dataset is further divided using a stratified split (preserving class distribution) with 10% allocated to a Validation set.
- The final data splits are: Train, Validation, and Test.

## 8. Preprocessing
- Images are decoded as JPEG with 3 channels and resized to 224x224.
- Integer labels are converted to one-hot encoding for categorical crossentropy loss.
- Class imbalance is mitigated by computing balanced class weights using `compute_class_weight` from scikit-learn.
- Architecture-specific preprocessing is applied (VGG16 preprocessing and MobileNetV2 preprocessing).

## 9. Data Augmentation
- Random flip left-right.
- Random brightness adjustment (max_delta=0.15).
- Random contrast adjustment (lower=0.85, upper=1.15).

## 10. TensorFlow Data Pipeline
- `tf.data.Dataset` is used to build pipelines from tensor slices of file paths and labels.
- Images are loaded, decoded, resized, and preprocessed in parallel using `tf.data.AUTOTUNE`.
- The dataset is cached after resizing but before augmentation to optimize performance.
- Augmentation is mapped, and the data is batched (batch_size=32) and prefetched.

## 11. Baseline Models
- Two baseline models are constructed:
  - **VGG16 Baseline**: Pre-trained on ImageNet with the backbone frozen.
  - **MobileNetV2 Baseline**: Pre-trained on ImageNet with the backbone frozen (training=False to freeze Batch Normalization statistics).
- Both use an identical Custom Classification Head:
  - GlobalAveragePooling2D
  - Dropout (0.3)
  - Dense (256 units, ReLU activation)
  - Dropout (0.3)
  - Dense (4 units, Softmax activation)

## 12. Training Strategy
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Class Weights**: Applied during training to handle class imbalance.
- **Callbacks**:
  - EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)
  - ModelCheckpoint (save_best_only=True)
  - ReduceLROnPlateau (monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
  - CSVLogger

## 13. Evaluation Methodology
- Best weights are restored from checkpoints.
- Models are evaluated on the unseen Test Dataset to compute Test Loss and Test Accuracy.
- A high-resolution, publication-quality Confusion Matrix is generated.
- A Classification Report is generated (Precision, Recall, F1-Score, Support).
- A Multi-Class ROC Curve (One-vs-Rest) is plotted, computing Micro-average, Macro-average, and class-specific ROC AUC.

## 14. Error Analysis
- Misclassified samples are isolated and rendered in a grid displaying ground-truth labels, predicted labels, and confidence percentages.
- Inter-class confusion frequencies are mapped to identify the most frequent overlaps.
- Absolute error rates per class and the average confidence when the model is wrong are calculated to profile misclassifications by true class.

## 15. Model Improvement
- **VGG16 Fine-Tuning**: The top convolutional block (Block 5) of the VGG16 base model is unfrozen, allowing the model to adapt to medical imaging features. All layers before Block 5 remain frozen.
- The model is recompiled with a micro-learning rate (Adam, 1e-5) to prevent catastrophic forgetting.
- Fine-tuning is executed for up to 15 epochs using dedicated callbacks (EarlyStopping with patience=4, ReduceLROnPlateau with patience=2).

## 16. Final Model Selection
- **VGG16 Fine-Tuned** is designated as the winner via an automated multi-metric selection.
- It outperformed the baselines by maximizing the Macro F1-Score (0.9638) and ROC AUC (0.9959).

## 17. Gradio Demonstration
- A user-friendly interface is implemented using Gradio.
- The application accepts a 2D MRI slice image upload, resizes it to 224x224, applies VGG16 preprocessing, and outputs the diagnostic probability for the 4 classes.

## 18. Final Outcome
- The project successfully established an end-to-end deep learning pipeline.
- The Fine-Tuned VGG16 model achieved a 96.38% Macro F1-Score and a 0.9959 Macro ROC AUC on unseen test data.
- Production artifacts (model weights and metadata) were successfully packaged.

## 19. Limitations
- **Parameter Intensity**: VGG16 requires significantly more memory and storage compared to MobileNetV2.
- **2D Slice Analysis**: The model processes independent 2D slices, which does not capture the full 3D volumetric context of the tumor.

## 20. Future Scope
- Transition to 3D Volumetric CNNs (e.g., 3D ResNets or DenseNets) to natively capture volumetric spatial context.
- Semantic Segmentation Integration using U-Net-based frameworks to perform pixel-level tumor localization.
- Multi-Modal Data Fusion to ingest multiple MRI sequences (T1, T2, and FLAIR) simultaneously.
