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

# 📊 Unemployment Analysis in India

##  Overview

This project performs **Exploratory Data Analysis (EDA)** on unemployment data from India to identify regional and temporal trends and analyze the impact of the **COVID-19 pandemic** on unemployment.

The project uses Python to clean, analyze, visualize, and interpret unemployment, employment, and labour participation data across different Indian states.

---

##  Objectives

* Analyze unemployment rates across Indian states.
* Identify states with the highest average unemployment.
* Study monthly unemployment trends.
* Compare unemployment before and after COVID-19.
* Analyze changes in employment and labour participation.
* Identify states most affected by the COVID-19 shock.
* Study correlations between major labour-market indicators.
* Measure unemployment volatility across regions.

---

##  Dataset

**Dataset:** Unemployment in India
**Source:** Kaggle
**Author:** Gokul Raj Kuppan

🔗 **Kaggle Dataset:**
https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india

The dataset contains state-wise and monthly information including:

* Region
* Date
* Frequency
* Estimated Unemployment Rate (%)
* Estimated Employed
* Estimated Labour Participation Rate (%)
* Area

> **Note:** The dataset provides `Estimated Employed`, not a direct `Employment Rate (%)`. Therefore, `Estimated Employed` is used as the employment indicator in this analysis.

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

##  Analysis Performed

### 1. Data Cleaning

* Loaded and inspected the dataset.
* Checked dataset dimensions and data types.
* Identified missing values.
* Checked duplicate records.
* Cleaned column names and categorical values.
* Converted the date column into datetime format.

### 2. Feature Engineering

Created additional features including:

* Year
* Month
* Month Name
* COVID Period
* COVID Shock

### 3. State-wise Analysis

Calculated state-wise:

* Average unemployment rate
* Average estimated employment
* Average labour participation rate

The **Top 10 states/regions with the highest average unemployment** were visualized using a bar chart.

### 4. Time-Series Analysis

Analyzed monthly unemployment trends to identify:

* Long-term patterns
* Major fluctuations
* Peak unemployment periods
* Changes during the COVID-19 period

Selected major states were also compared using time-series line charts.

### 5. COVID-19 Impact Analysis

The data was divided into relevant COVID periods to compare:

* Unemployment Rate
* Estimated Employed
* Labour Participation Rate

This helps quantify how labour-market conditions changed during the pandemic.

### 6. Correlation Analysis

A correlation heatmap was created to examine relationships between:

* Unemployment Rate
* Estimated Employed
* Labour Participation Rate

### 7. Volatility Analysis

Standard deviation was used to measure unemployment volatility across regions.

This identifies states where unemployment experienced the largest fluctuations over time.

### 8. Labour Participation vs Unemployment

A scatter plot with a regression trend was used to examine the relationship between labour participation and unemployment.

---

##  Visualizations

The project includes:

*  Overall unemployment trend
*  State-wise unemployment trends
*  Top 10 states by average unemployment
*  COVID-19 impact comparison
*  Correlation heatmap
*  Unemployment volatility analysis
*  Labour participation vs unemployment
*  State-wise COVID impact

---

##  Key Insights

The analysis identifies:

* Regions with the highest average unemployment.
* Periods of peak unemployment.
* Changes in unemployment during COVID-19.
* Changes in estimated employment and labour participation.
* States experiencing the greatest unemployment volatility.
* Relationships between labour participation and unemployment.

The exact numerical findings are generated directly from the dataset when the notebook is executed.

---



##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/soup511/unemployment-analysis-india.git
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### 3. Download the dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india

Place the CSV file in the appropriate project directory.

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

```text
Unemployment_Analysis_India.ipynb
```

and run all cells sequentially.

---

##  Limitations

* The dataset covers a limited time period.
* `Estimated Employed` represents an employment estimate rather than an employment-rate percentage.
* State-level averages may hide rural/urban differences.
* Correlation does not imply causation.
* COVID-19 unemployment changes may also reflect other economic factors occurring during the same period.

---

##  Future Improvements

Possible extensions include:

* Adding newer unemployment data.
* Rural vs urban unemployment analysis.
* Gender-wise unemployment analysis.
* Interactive dashboards using **Power BI or Tableau**.
* Statistical hypothesis testing.
* Unemployment forecasting.
* Machine-learning-based unemployment prediction.

---

##  Author

**Souptika Das**

B.Tech Information Technology
RCC Institute of Information Technology

# 🚗 Car Price Prediction with Machine Learning

##  Project Overview

This project develops a **Machine Learning regression system to predict the selling price of used cars** based on vehicle characteristics and seller information.

The project follows an end-to-end machine learning workflow:

**Data Loading → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Encoding → Model Training → Model Evaluation → Feature Analysis → Price Prediction**

The objective is to understand the factors influencing used-car prices and build a model capable of estimating the selling price of a vehicle.

---

##  Objectives

* Analyze the characteristics of used cars.
* Understand factors affecting selling prices.
* Clean and preprocess the dataset.
* Engineer meaningful features from existing data.
* Encode categorical variables for machine learning.
* Train multiple regression models.
* Compare model performance using standard regression metrics.
* Identify important factors influencing car prices.
* Demonstrate price prediction on a sample vehicle.

---

##  Dataset

The project uses the **Car Details from CarDekho** dataset.

The dataset contains information about used cars, including:

| Feature         | Description                                |
| --------------- | ------------------------------------------ |
| `name`          | Name/model of the car                      |
| `year`          | Manufacturing year                         |
| `selling_price` | Selling price of the car — target variable |
| `km_driven`     | Kilometres driven                          |
| `fuel`          | Fuel type                                  |
| `seller_type`   | Type of seller                             |
| `transmission`  | Transmission type                          |
| `owner`         | Ownership history                          |

The original dataset contains **4,340 records and 8 columns**.

---

##  Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

---

##  Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas and initially contains:

* **4,340 rows**
* **8 columns**

The dataset includes numerical variables such as `year`, `selling_price`, and `km_driven`, along with categorical variables such as fuel type, seller type, transmission, and owner.

---

### 2. Data Cleaning

The dataset was examined for:

* Missing values
* Duplicate records
* Data types
* Categorical inconsistencies

No missing values were found in the dataset.

A total of **763 duplicate records** were identified and removed.

After removing duplicates, the dataset contained:

**3,577 records and 8 columns.**

---

##  Feature Engineering

Two additional features were created to improve the model:

###  Car Age

Car age was derived from the manufacturing year:

```text
Car_Age = Current Year - Manufacturing Year
```

This provides a more intuitive representation of vehicle depreciation.

###  Brand

The vehicle brand was extracted from the first word of the car name.

For example:

```text
Maruti 800 AC → Maruti
Hyundai Verna 1.6 SX → Hyundai
Honda Amaze VX i-DTEC → Honda
```

This allows the model to capture differences between car manufacturers.

---

#  Exploratory Data Analysis

The project explores how different vehicle characteristics relate to selling price.

The analysis includes visualizations for factors such as:

* Car age
* Mileage
* Fuel type
* Transmission
* Seller type
* Ownership
* Vehicle brand

For example, a boxplot is used to compare selling prices across different transmission types.

These visualizations help identify pricing patterns and potential outliers before model training.

---

#  Machine Learning Models

Three regression algorithms were trained and compared.

### 1. Linear Regression

A baseline regression model used to establish a simple relationship between vehicle characteristics and selling price.

### 2. Random Forest Regression

An ensemble tree-based model capable of capturing nonlinear relationships between vehicle features and price.

### 3. Gradient Boosting Regression

A boosting-based ensemble method that sequentially improves predictions by learning from previous errors.

Categorical features were converted using **OneHotEncoder**, implemented through a Scikit-learn `ColumnTransformer` and `Pipeline`.

---

#  Model Evaluation

The models were evaluated using three standard regression metrics:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

**Lower MAE = better performance**

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more heavily.

**Lower RMSE = better performance**

### R² Score

Measures how much of the variation in selling price is explained by the model.

**Higher R² = better performance**

The project compares all three models using these metrics.

---

##  Model Comparison

The notebook generates a model-performance comparison using R² scores.

The evaluated models achieve R² scores in the approximate range of:

| Model             |     R² Score |
| ----------------- | -----------: |
| Linear Regression | ~0.421–0.589 |
| Random Forest     | ~0.421–0.589 |
| Gradient Boosting | ~0.421–0.589 |

The exact model-to-score mapping should be taken from the final `results` table generated when the notebook is executed. The notebook selects the model with the **highest R² score as the best-performing model**.

---

#  Feature Importance

Feature importance analysis is used to understand which vehicle characteristics have the greatest influence on predicted selling prices.

This makes the project more interpretable by going beyond simply producing predictions.

The analysis helps answer questions such as:

* Does vehicle age strongly affect price?
* How important is mileage?
* Does brand influence resale value?
* Does transmission type affect pricing?
* How does ownership history contribute to price?

---

#  Example Prediction

The trained Random Forest model is used to predict the selling price of a sample vehicle with characteristics such as:

```text
Kilometres Driven : 30,000
Fuel              : Petrol
Seller Type       : Dealer
Transmission      : Manual
Owner             : First Owner
Car Age           : 5 years
Brand             : Toyota
```

The notebook produces an example predicted selling price of approximately:

**₹9,39,528**

---

#  Key Results

The project demonstrates that used-car prices can be modeled using a combination of:

* Vehicle age
* Mileage
* Brand
* Fuel type
* Seller type
* Transmission
* Ownership history

Multiple regression algorithms were trained and evaluated using MAE, RMSE, and R².

The model with the highest R² score is selected as the final model, followed by feature-importance analysis and an example price prediction.

---

#  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/soup511/OIBSIP.git
```

### 2. Open the notebook

Open:

```text
SouptikaDas_Task3.ipynb
```

using **Google Colab** or **Jupyter Notebook**.

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 4. Add the dataset

Place:

```text
CAR DETAILS FROM CAR DEKHO.csv
```

in the appropriate directory.

### 5. Run the notebook

Execute the cells sequentially from data loading through model evaluation and prediction.

---

#  Machine Learning Concepts Demonstrated

This project demonstrates practical knowledge of:

* Exploratory Data Analysis
* Data Cleaning
* Duplicate Detection
* Feature Engineering
* Categorical Encoding
* One-Hot Encoding
* Train-Test Split
* Machine Learning Pipelines
* Regression
* Linear Regression
* Random Forest Regression
* Gradient Boosting
* Model Comparison
* MAE
* RMSE
* R² Score
* Feature Importance
* Predictive Modeling

---

# ⚠️ Limitations

* The dataset contains information about used cars available in the underlying CarDekho data and may not represent the entire used-car market.
* Selling prices can be influenced by factors not included in the dataset, such as vehicle condition, location, service history, accident history, and additional features.
* The dataset's `name` field contains detailed model information, while the engineered `Brand` feature only extracts the first word.
* Model performance depends on the available features and the train-test split.

---

# 🔮 Future Improvements

The project could be extended by:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Adding more detailed car-model features.
* Applying cross-validation.
* Testing XGBoost or other advanced boosting models.
* Using SHAP for model explainability.
* Building an interactive car-price prediction web application.
* Deploying the final model using Streamlit or FastAPI.
* Adding location, vehicle condition, service history, and insurance information.

---

#  Author

**Souptika Das**

B.Tech Information Technology
RCC Institute of Information Technology

