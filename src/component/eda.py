# import necessary libraries
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
import prince
from collections import Counter
import re

# set random seed for reproducibility
np.random.seed(42)

# configuration for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================
print("--- Setting up Paths ---")
# ==============================================

# path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..','data', 'combined')
results_dir = os.path.join(script_dir, '..', 'results', 'EDA')

# create the results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")
else:
    print(f"Directory already exists: {results_dir}")

# ==============================================
print("--- Loading Data ---")
# ==============================================

try:
    prod_path = os.path.join(data_dir, 'products.csv')
    products = pd.read_csv(prod_path)
    print("Products data loaded successfully.")

    reviews_path = os.path.join(data_dir, 'reviews.csv')
    reviews = pd.read_csv(reviews_path, na_values=['', 'NA', 'NaN'])
    print("Reviews data loaded successfully.")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# ==============================================
print(" --- Cleaning and Preprocessing ---")
# ==============================================

# rename columns
products = products.rename(columns={
    "brand": "Brand", "key": "ProductID", "name": "ProductName",
    "subhead": "Subhead", "description": "ProductDescription",
    "rating": "Rating", "rating_count": "RatingCount", "ingredients": "Ingredients"
})

reviews = reviews.rename(columns={
    "brand": "Brand", "key": "ProductID", "author": "Username",
    "date": "Date", "stars": "Stars", "title": "Title",
    "helpful_yes": "Helpful", "helpful_no": "Not Helpful",
    "text": "Review", "taste": "Taste", "ingredients": "IngredientsDesc",
    "texture": "Texture", "likes": "Likes"
})

# feature engineering
def create_flag(df, column, pattern):
    return df[column].astype(str).str.contains(pattern, case=False, na=False).astype(int)

products['Has_Vanilla'] = create_flag(products, 'Ingredients', 'Vanilla')
products['Has_Organic'] = create_flag(products, 'Ingredients', 'Organic')
products['Has_Caramel'] = create_flag(products, 'Ingredients', 'Caramel')
products['Has_Chocolate'] = create_flag(products, 'Ingredients', 'Chocolat|Coco')
products['Has_Fruit'] = create_flag(products, 'Ingredients', 'Raspberr|Cherr|Blueberr|Banana')

is_bar_name = products['ProductName'].astype(str).str.contains('\bBar\b', case=False, regex=True)
is_bar_sub = products['Subhead'].astype(str).str.contains('\bBar\b', case=False, regex=True)
products['Is_Bar'] = (is_bar_name | is_bar_sub).astype(int)

