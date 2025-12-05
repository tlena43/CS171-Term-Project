# CS171 Term Project Read Me

## Project Title
**Machine Learning for Waste Processing: Using Convolutional Neural Networks for Smarter Recycling and E-Waste Classification**

---
## Presentation
Our final project presentation can be viewed here:

👉 **[View the Presentation Slides][(https://docs.google.com/presentation/d/1SaytfMzCnZ6yq7Ptjw1ku0B6XT7OS7QDqZTWZM_4ecc/edit?usp=sharing)]**  

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


---
## Description of Question and Research Topic
Electronic and household waste are growing environmental challenges that require smarter sorting and recycling systems. Our project explores how convolutional neural networks (CNNs) can classify different types of waste images to improve automated recycling efficiency. Samriddhi focuses on identifying ten types of **electronic waste** (e.g., TV, battery, circuit board), while Helena focuses on **recyclables** (e.g., aluminum cans, glass bottles). By comparing model performance and confusion matrices, we aim to determine how computer vision can support real-world waste management pipelines. This work demonstrates the role of machine learning in promoting sustainability and reducing landfill impact.

---

## Research Question

**How effectively can modern deep learning models classify different types of waste, and do these models generalize well enough to support real-world automated recycling systems?**
---

## Project Outline

### Data Collection

### Samriddhi Matharu
- **Dataset:** *E-Waste Image Dataset* (~3,000 labeled images across 10 classes: Battery, Mobile Phone, Mouse, PCB, Printer, Player, Television, Washing Machine, etc.) 
- Data is pre-organized into `train/val/test` folders from Kaggle.
- Merged Kaggle’s validation split into the test set to form a larger and more reliable test dataset.
- Preprocessing steps include loading images using `ImageFolder`.  
- Resize images, convert to tensors, and normalize pixel values.  
- Apply light augmentation (flip, rotation) to increase variety and help prevent overfitting.
- Created an independent, real-world validation folder (“val by hand”) with unseen images collected manually from the internet.

### Helena Thiessen
- **Dataset:** *Drinking Waste Classification Dataset* (~4000 labeled images across 4 classes).
- Research datasets pertaining to recycling and select one of interest.
- Create a custom dataset object
- Data must be read into python and stored as tensors.
- Labels must be read into python and properly processed.
- Using pytorch data must then be normalized, and augmented to reduce overfitting.
- Split data into test and train sets.
- Independently create a validation dataset

---

### Model Plans

### Samriddhi Matharu
-  Build and compare two models for 10-class e-waste classification.  
- Implemented a **custom Convolutional Neural Network (CNN)** as a baseline model.  
- Designed the CNN with multiple convolutional layers, dropout, and standard image augmentations.  
- Used this model to establish a starting point for understanding the complexity of the dataset.
- Implemented a **pretrained ResNet18** using ImageNet weights to explore transfer learning.  
- Froze the backbone layers and fine-tuned the final layer for 10-class classification.  
- Planned to compare the behavior of a model trained from scratch versus a pretrained model on both the Kaggle dataset and a small curated validation set done by hand on rougher images

### Helena Thiessen
- Use a **Region Based Convolution Neural Network (R-CNN)** to detect the presence of recycling items of given classes.
- Research R-CNN's in pytorch
- Iplement a custom backbone
- Use custom backbone with pytorch FasterRCNN object
- Finetune backbone by trying different structures
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

### How to Run

### Helena Thiessen
- Install git lfs
    -  model.pt is over the size limit for normal file storage with git and is handled through git lfs
- Clone git repository
    - Note: folder and path structure must be maintained for the code to locate files
- Install required libraries
    - Use command `pip install -r requirements.txt` to install libraries in your environment
    - Note that torch installs may vary between machines depending on CUDA availability
    - Alternatively install only the packages you are missing by using command `pip install ~name~`
- No special instructions are required for accessing data because it is all within the repository under `HelenaT_Project/Images_of_Waste`
- Provided file structure is maintained from github repository all of my parts are ready to run
- Open `HelenaT_Project` within `CS171-Term-Project`
- `preprocess.ipynb` contains code relevant to preprocessing data
- `R-CNN.ipynb` contains the code relevant to creating and training the model
- To view analysis, open `03_analysis_visualization_ewaste` in `CS171-Term-Project`
    - The analysis and visualization of my RCNN are under Section 2

---

### How to Run

### Samriddhi Matharu

- **Clone the GitHub repository**
  - Make sure the folder and path structure remain unchanged so notebooks can locate files correctly.

- **Install required libraries**
  - Run: `pip install -r requirements.txt`
  - PyTorch installation may vary depending on CUDA availability.
  - If needed, install missing packages individually using: `pip install <package_name>`

- **Download the dataset**  
  - The E-Waste dataset is **not stored in the repository** due to size limits.  
  - Follow the instructions in `SamriddhiM_Project/data/README_DATA.md` to download the Kaggle dataset and place it in the correct folders.

- **Notebook order**
  1. **`01_data_preprocessing_ewaste.ipynb`** — Loads and preprocesses dataset.  
  2. **`02_model_training_ewaste.ipynb`** — Trains the custom CNN and pretrained ResNet18 models.  
  3. **`03_analysis_visualization_ewaste.ipynb`** — Runs metrics, confusion matrices, and analysis (Samriddhi's analysis is in Section 1).

---

### Further Works

### Samriddhi Matharu
Future updates and extensions to the E-Waste classification project could include:

- Expanding the dataset with more diverse, real-world images to improve generalization.  
- Adding stronger augmentations (color jitter, random crops, noise) during training.  
- Exploring deeper architectures such as ResNet34, ResNet50, or EfficientNet.  
- Re-train model completely on diverse data found by hand isntead of kaggle
- Deploying a lightweight model to simulate real-time recycling or sorting operations.
- Scale and make this an application 

---

### Further Works

### Helena Thiessen
Updates to the Drinking Waste RCNN should include:
- Source new images that can be added to the dataset for more diverse training data
- Annotate bounding boxes for new images
- Split the dataset into distinct Train/Test/Validation folders to ensure a consistent split since the dataset may no longer be class proportional and images of all types are desired in all sets
- Retrain the existing model on the more diverse dataset
- Assess model results to determine next steps

---

Licensed under Apache 2.0 license
