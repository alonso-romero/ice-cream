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