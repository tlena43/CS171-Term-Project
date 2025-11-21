# CS171 Term Project Read Me

## Project Title
**Machine Learning for Waste Processing: Using Convolutional Neural Networks for Smarter Recycling and E-Waste Classification**

---

## Authors
- **Samriddhi Matharu** — E-Waste Image Classification  
- **Helena Thiessen** — Recycling and Trash Image Classification

---

## Data Sources
- **E-Waste Image Dataset** — [Kaggle: E-Waste Image Dataset](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset/data)  
  Used by *Samriddhi Matharu* for classifying 10 categories of electronic waste items.  

- **WARP Dataset** — [Kaggle: Waste Recycling Plant Dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset/data)  
  Used by *Helena Thiessen* for classifying between different types of recyclables.

- **TACO Dataset** — [Taco Dataset](http://tacodataset.org/)  
  Used by *Helena Thiessen* for classifying between different types of trash and recyclables. 

---
## Description of Question and Research Topic
Electronic and household waste are growing environmental challenges that require smarter sorting and recycling systems. Our project explores how convolutional neural networks (CNNs) can classify different types of waste images to improve automated recycling efficiency. Samriddhi focuses on identifying ten types of **electronic waste** (e.g., TV, battery, circuit board), while Helena focuses on **general waste and recyclables** (e.g., plastic, glass, organic). By comparing model performance and confusion matrices, we aim to determine how computer vision can support real-world waste management pipelines. This work demonstrates the role of machine learning in promoting sustainability and reducing landfill impact.

---
## Project Outline

### Data Collection

### Samriddhi Matharu
- **Dataset:** *E-Waste Image Dataset* (~3,000 labeled images across 10 classes, Apache 2.0 License).  
- Data is pre-organized into `train/val/test` folders from Kaggle.
- Merged Kaggle’s validation split into the test set to form a larger and more reliable test dataset. 
- Preprocessing steps include loading images using `ImageFolder`.  
- Resize images, convert to tensors, and normalize pixel values.  
- Apply light augmentation (flip, rotation) to increase variety and help prevent overfitting.
- Created an independent, real-world validation folder (“val by hand”) with unseen images collected manually from the internet.

### Helena Thiessen
- **Dataset:** *Waste Recycling Plant Dataset* (~3000 labeled images across 17 classes).
- **Dataset:** *TACO Dataset* (~1500 labelled images across 60 classes).
- Research datasets pertaining to recycling and select one of interest.
- Decide between or combine TACO and WARP based on usability
- The WARP data is split into training and validation sets.
- The TACO data is not split
- Some categories may need to be combined or removed to narrow down to the desired categories.
- Data must be read into python and stored as tensors.
- Labels must be read into python and properly processed.
- Using pytorch data must then be normalized, and augmented to reduce overfitting.
- Independently create a validation dataset

---

### Model Plans

### Samriddhi Matharu
- Use a **Convolutional Neural Network (CNN)** to classify e-waste images into 10 categories.    
- We plan to adjust the number of layers and filters to see how they affect performance.  
- Built a custom CNN with two convolutional blocks and dropout, trained for 30 epochs with augmentation.
- Evaluated using accuracy, loss curves, confusion matrix, and per-class precision/recall/F1.
- Implemented a pre-trained ResNet18 model, froze base layers, and fine-tuned the final layer for 10 classes.
- Compared CNN vs. ResNet18 on: Kaggle test set and Custom real-world validation set
- Analyzed generalization differences, where the CNN struggled on real-world images, while ResNet18 performed much better


### Helena Thiessen
- Use a **Region Based Convolution Neural Network (R-CNN)** to detect the presence of recycling items of given classes.
- Research R-CNN's in pytorch
- Iplement a custom backbone
- Use custom backbone with pytorch FasterRCNN object
- Finetune backbone with varying arrangements of pooling layers, convolutional layers, linear layers, and dropout layers
- Finetune model by trying different activation functions and gradient descent algorithms
- Train model and assess results on test data

---

### Project Timeline
- Week 9: Introduce research topic and timeline
- Week 10: Source and prepare data
- Week 11: Data preprocessing
- Week 12: Begin designing models
- Week 13: Fine tune models
- Week 14: Assess results
- Week 15: Perform analysis and prepare presentation
- Week 16: Present project
- Week 17: Submit Project

---
Licensed under Apache 2.0 license
