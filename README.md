# OIBSIP
# 🌸 Iris Flower Classification Using Machine Learning
This project is a small machine learning classification project based on the well-known Iris dataset. The idea is simple: given the measurements of an iris flower, the model tries to determine which species it belongs to.
The prediction is made using four measurements:
* Sepal length
* Sepal width
* Petal length
* Petal width
The possible predictions are **Setosa, Versicolor, and Virginica**.
I worked through the project step by step, starting with understanding the dataset and ending with a comparison of different classification algorithms.

##  What I Wanted to Achieve
The main goal was to understand how a classification problem works in practice.
For this project, I:
* Explored the dataset before training any model
* Checked the data for missing values and basic errors
* Used graphs to understand the differences between species
* Identified which measurements were more useful for classification
* Trained multiple machine learning algorithms
* Compared their performance using different evaluation metrics
* Selected a preferred model based on the results

##  Tools and Libraries
The project was developed using **Python** and a Jupyter Notebook.
The main libraries used were:
| Library          | Purpose                         |
| ---------------- | ------------------------------- |
| NumPy            | Numerical operations            |
| Pandas           | Working with the dataset        |
| Matplotlib       | Creating plots                  |
| Seaborn          | Statistical visualizations      |
| Scikit-learn     | Machine learning and evaluation |
| Jupyter Notebook | Writing and running the project |

##  About the Dataset
I used the Iris dataset that comes directly with **scikit-learn**, so no separate dataset file was required.

The dataset contains **150 flower samples** divided equally among three species. Each sample has four numerical measurements.

### Input Features

| Feature      | Unit |
| ------------ | ---- |
| Sepal Length | cm   |
| Sepal Width  | cm   |
| Petal Length | cm   |
| Petal Width  | cm   |

### Target
The target variable represents the flower species:
* Setosa
* Versicolor
* Virginica

##  Exploring the Data
Before building the models, I performed some basic exploratory analysis.
I checked:
* Number of rows and columns
* Data types
* Missing values
* Summary statistics
* General information about the DataFrame
The dataset contains **150 rows and 5 columns**, and there are **no missing values**.

##  Visual Analysis
I used two main types of plots to understand the data.
### Pairplot
The pairplot helped me see how the four measurements relate to one another and how the three species are distributed.
One of the clearest observations was that the **petal measurements separate the species much better than the sepal measurements**.

### Box Plots
Box plots were used to compare the distribution of every measurement across the three species.
These plots supported the observation that **petal length and petal width are especially useful features** for identifying the species.

##  Feature Selection
After looking at the plots and feature correlations, I found that **petal length and petal width provide the clearest separation between the species**.

Setosa is particularly easy to distinguish from the other two species using these measurements. Versicolor and Virginica are somewhat closer to each other, so their classification is more challenging.

Although some features appear more informative than others, I kept all four measurements when training the models because the dataset is small and the remaining features may still contribute useful information.

##  Preparing the Data
The features were separated from the target variable, and the dataset was divided into training and testing portions.

I used an **80:20 split**, meaning that 80% of the samples were used for training and the remaining 20% were kept aside for testing.

A fixed random state was used so that the same split can be reproduced when the notebook is run again.

##  Models Tested
I experimented with three different classification algorithms.

### 1. Logistic Regression
This was used as a simple classification approach. It is relatively easy to understand and interpret, making it a useful model for comparison.

### 2. K-Nearest Neighbours
KNN makes a prediction by looking at nearby training examples. I used **5 neighbours** for this experiment.

### 3. Decision Tree
The Decision Tree learns a set of feature-based rules and uses those rules to decide which species a flower belongs to.

Using three models allowed me to compare different approaches instead of depending on a single algorithm.

##  How the Models Were Evaluated
I used several metrics to check the predictions:
* **Accuracy** – percentage of correct predictions
* **Precision** – how reliable the positive predictions are
* **Recall** – how many samples of a class were correctly identified
* **F1-score** – a combined measure of precision and recall
* **Confusion Matrix** – shows the actual and predicted classes

##  Results
For the particular 80/20 train-test split used in this notebook, all three models produced the same test accuracy:
| Model                | Test Accuracy |
| -------------------- | ------------: |
| Logistic Regression  |          100% |
| K-Nearest Neighbours |          100% |
| Decision Tree        |          100% |
The classification reports also showed perfect precision, recall, and F1-score for the test samples.
Since the models had identical test accuracy, I did not choose the model simply because of its accuracy. I preferred **Logistic Regression** because it provides the same observed test performance with a simpler and more interpretable model.

##  What I Learned
This project helped me understand the practical flow of a classification problem.
Some of the main things I learned were:
* How to load a built-in dataset using scikit-learn
* How to inspect a dataset before modelling
* How visualizations can reveal useful features
* How to divide data into training and testing sets
* How different classification algorithms can be compared
* How confusion matrices and classification reports are used to evaluate predictions
* Why model selection should consider more than just one metric

##  Conclusion

The Iris dataset turned out to be well suited for demonstrating a basic classification workflow. The exploratory analysis showed that petal length and petal width are particularly helpful in distinguishing the three species.
After training Logistic Regression, KNN, and Decision Tree models, all three performed perfectly on the selected test set. Based on the identical test performance and its simpler structure, **Logistic Regression was chosen as the preferred model for this project**.
This project gave me practical experience with the complete process of preparing data, exploring it, training classification models, evaluating their predictions, and comparing the results.

##  Running the Project
To run the notebook locally, install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

Then open the `.ipynb` file using **Jupyter Notebook, JupyterLab, or Google Colab** and run the cells from beginning to end.
````
## Author
Souptika Das
B.Tech – Information Technology

