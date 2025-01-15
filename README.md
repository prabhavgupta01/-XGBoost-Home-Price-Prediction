# -XGBoost-Home-Price-Prediction
# House Price Prediction and Analysis

This project focuses on exploratory data analysis (EDA) and predictive modeling for house prices using the `kc_house_data.csv` dataset. It utilizes Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn to analyze and visualize data, and to build machine learning models for predicting house prices.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Analysis](#data-analysis)
- [Predictive Modeling](#predictive-modeling)
- [Results](#results)
- [Usage](#usage)

---

## Project Overview
The project includes:
1. Data visualization to explore relationships between features.
2. Training a Linear Regression model.
3. Using Gradient Boosting Regressor for improved predictions.
4. Evaluating model performance.

---

## Dataset
The dataset contains information about house sales in King County, USA, including features such as:
- Number of bedrooms and bathrooms
- Square footage
- Geographic coordinates (latitude and longitude)
- Price
- Waterfront presence

### Sample Features
- **price**: Price of the house.
- **bedrooms**: Number of bedrooms.
- **sqft_living**: Square footage of the house.
- **waterfront**: Indicates if the property is by the waterfront.
- **date**: Date of sale.

---

## Dependencies
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

To install the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Data Analysis
Key visualizations and analyses:
1. **Bedroom Counts**:
   - Bar plot to visualize the distribution of bedrooms.
2. **Price vs. Features**:
   - Scatter plots for price vs. square footage, latitude, and waterfront.
3. **Geographic Insights**:
   - Joint plots for latitude and longitude to analyze location trends.

### Example
```python
plt.scatter(data.price, data.sqft_living)
plt.title("Price vs Square Feet")
plt.xlabel("Price")
plt.ylabel("Square Feet")
plt.show()
```

---

## Predictive Modeling
### Linear Regression
- Built using `scikit-learn`'s `LinearRegression`.
- Data split into training and testing sets using `train_test_split`.

### Gradient Boosting Regressor
- Model trained with 400 estimators, max depth of 5, and a learning rate of 0.1.
- Evaluated using the test set.

### Example Code
```python
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(n_estimators=400, max_depth=5, learning_rate=0.1, loss='squared_error')
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(f"Model Score: {score}")
```

---

## Results
- Linear Regression and Gradient Boosting Regressor were implemented.
- Gradient Boosting Regressor outperformed Linear Regression in terms of prediction accuracy.

---

## Usage
1. Ensure the dataset (`kc_house_data.csv`) is in the same directory as the script.
2. Run the script in a Python environment.
3. Modify parameters like `n_estimators` and `learning_rate` for experimentation.

---

## Future Improvements
- Feature engineering for additional insights.
- Hyperparameter tuning using Grid Search or Randomized Search.
- Incorporating other machine learning algorithms like Random Forests or XGBoost.

---

## License
This project is open-source and available under the MIT License.

