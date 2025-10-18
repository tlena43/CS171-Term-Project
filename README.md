# CS171 Term Project Read Me

## Project Title
**Machine Learning for Waste Processing: Using Convolutional Neural Networks for Smarter Recycling and E-Waste Classification**

---

## Authors
- **Samriddhi Matharu** — E-Waste Image Classification  
- **Helena Thiessen)** — FILL IN

---

## Data Sources
- **E-Waste Image Dataset** — [Kaggle: E-Waste Image Dataset](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset/data)  
  Used by *Samriddhi Matharu* for classifying 10 categories of electronic waste items.  

- **DATASET NAME** — [Name Your URL Here ] (link)  
  Used by *Helena Thiessen* for ..

---
## Description of Question and Research Topic
Electronic and household waste are growing environmental challenges that require smarter sorting and recycling systems. Our project explores how convolutional neural networks (CNNs) can classify different types of waste images to improve automated recycling efficiency. Samriddhi focuses on identifying ten types of **electronic waste** (e.g., TV, battery, circuit board), while Helena focuses on **general recyclable waste** (e.g., plastic, glass, organic). By comparing model performance and confusion matrices, we aim to determine how computer vision can support real-world waste management pipelines. This work demonstrates the role of machine learning in promoting sustainability and reducing landfill impact.

---
## Project Outline

### Data Collection

### Samriddhi Matharu
- **Dataset:** *E-Waste Image Dataset* (~3,000 labeled images across 10 classes, Apache 2.0 License).  
- Data is pre-organized into `train/val/test` folders.  
- Preprocessing steps include loading images using `ImageFolder`.  
- Resize images, convert to tensors, and normalize pixel values.  
- Apply light augmentation (flip, rotation) to increase variety and help prevent overfitting.

### Helena Thiessen
FILL IN

---

### Model Plans

### Samriddhi Matharu
- Use a **Convolutional Neural Network (CNN)** to classify e-waste images into 10 categories.  
- The model will include convolution, ReLU activation, and pooling layers to extract features, followed by fully connected layers for classification.  
- We plan to adjust the number of layers and filters to see how they affect performance.  
- Evaluate results using accuracy and loss curves, and visualize model predictions with a confusion matrix.

### Helena Thiessen
FILL IN

---

### Project Timeline

---
Licensed under Apache 2.0 license
