# Titanic Survival Prediction - Data Preprocessing

This project focuses on **data cleaning, handling null values, encoding categorical features, and feature scaling** using the Titanic dataset.  
The preprocessing ensures the dataset is ready for applying Machine Learning models.

---

## 📂 Dataset Overview
- **Source:** Titanic dataset (Kaggle)
- **Rows:** 891
- **Columns:** 12  
  (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

---

## 🔧 Preprocessing Steps

### 1. Handling Missing Values
- **Age:** Missing values filled with the **median**.
- **Cabin:** Dropped (too many missing entries).
- **Embarked:** Missing values filled with the **mode**.

### 2. Encoding Categorical Features
- **Sex:** Encoded (0 = female, 1 = male).
- **Embarked:** Encoded using label encoding.

### 3. Dropping Irrelevant Columns
- Removed **PassengerId, Name, Ticket** (not useful for prediction).

### 4. Feature Scaling
- Applied **StandardScaler** to numerical features to normalize data.

---

## 📊 Final Dataset
- Columns: `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived`
- Cleaned and ready for ML model training.

---

## ▶️ Example Usage
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Handle nulls
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Drop irrelevant columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Scale
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['Survived']))
