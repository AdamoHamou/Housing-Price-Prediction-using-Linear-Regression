# Housing Price Prediction using Linear Regression

This project implements a **Linear Regression model** using scikit-learn to predict housing prices based on multiple features from a Kaggle dataset. The goal is to evaluate how well linear regression can model real-world housing data and to demonstrate model evaluation techniques.

## 📊 Dataset
- **Source:** [Kaggle – House Price Prediction Treated Dataset](https://www.kaggle.com/datasets/aravinii/house-price-prediction-treated-dataset/data)  
- Features include: `price`, `bedrooms`, `grade`, `living_in_m2`, `real_bathrooms`, `month`, `quartile_zone`, `sale_year`, `sale_month`, and more.

## 🛠️ Methods
1. **Data Preprocessing**
   - Normalized numerical features using `StandardScaler` from `sklearn.preprocessing`.
   - Excluded boolean features from normalization to preserve categorical meaning.
   - Feature selection performed with `SelectKBest` to identify the 15 most predictive features.

2. **Model Training**
   - Implemented **LinearRegression** from `sklearn.linear_model`.
   - Trained on normalized data and evaluated using **Root Mean Squared Error (RMSE)**.

3. **Cross-Validation**
   - Applied **10-Fold Cross Validation** using `sklearn.model_selection.KFold`.
   - Analyzed coefficient stability across folds to validate model robustness.

4. **Visualization**
   - Used **Matplotlib** and **Seaborn** to plot predicted vs. actual prices.
   - Displayed regression line and scatterplot for visual correlation.

## 📈 Results
- Achieved consistent RMSE values across folds.
- Regression plots showed clear correlation between predicted and actual prices.
- Outliers were present at higher price ranges, as expected due to dataset complexity.

## 📚 Libraries Used
- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

## 🚀 Lessons Learned
- Importance of **feature scaling** and **selection** in regression performance.
- How to apply **K-Fold Cross Validation** to test model generalization.
- Experience with regression visualization and data preprocessing pipelines.

---

### 🔗 Link
Source code and notebook are available in this repository.
