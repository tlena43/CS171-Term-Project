📄 README_DATA.md

Dataset Access Instructions

The dataset used in this project is too large to upload to GitHub due to file-size limits.
To reproduce the results, please follow the steps below.

1. Download the E-Waste Dataset (Kaggle)
This project uses the E-Waste Image Classification Dataset from Kaggle:

Dataset link:
[https://www.kaggle.com/datasets/techsash/waste-classification-data](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset/data)

Download the dataset ZIP file from Kaggle and unzip it on your local machine.

📁 2. Create the Required Folder Structure

After unzipping, please create the following structure inside the project:

samriddhi/
   data/
      train/
      test/        <-- merged Kaggle test + Kaggle validation


If you are using the optional curated validation set, it also goes in the data folder 

 3. Preparing the Data Before Running the Notebooks
Training Data

Place the Kaggle train folder inside:

samriddhi/data/train/

Test Data

In the project, the Kaggle test and validation sets were merged manually.
To recreate this:

Combine both folders into a single folder.

Place it here:

samriddhi/data/test/

Optional: Curated Validation Set

If included, place the custom images here:

samriddhi/data/


These were manually collected to simulate real-world lighting, angle, and background variation.


4. Running the Project

Once the dataset folders are in place:

Run Notebook 1: Data Preprocessing
This notebook loads and prepares the dataset.

Run Notebook 2: Model Training
This trains CNN V3 and ResNet18 models.

Run Notebook 3: Analysis & Visualization
This evaluates model performance and displays results.
