# import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import itertools

# configuration for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def create_flag(df, column, pattern):
    return df[column].astype(str).str.contains(pattern, case=False, na=False).astype(int)


def run_eda():
    """
    Executes the Exploratory Data Analysis (EDA) process on ice cream product and review datasets.
    1. Sets up paths relative to the script location.
    2. Loads product and review data from CSV files.
    3. Cleans and preprocesses the data, including renaming columns and feature engineering.
    4. Generates and saves various graphs and visualizations to the results directory.
    5. Performs statistical analyses including normality tests and pairwise t-tests.
    6. Saves the cleaned datasets for downstream modeling.
    """

    # ==============================================
    print("\n--- Setting up Paths ---")
    # ==============================================

    # path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..','data', 'combined')
    results_dir = os.path.join(script_dir, '..', 'results', 'EDA')
    cleaned_data_dir = os.path.join(script_dir, '..', 'data', 'clean-data')

    # create the results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # create the clean data directory if it doesn't exist
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    # ==============================================
    print("\n--- Loading Data ---")
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
        return

    # ==============================================
    print("\n--- Cleaning and Preprocessing ---")
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

    products['Has_Vanilla'] = create_flag(products, 'Ingredients', 'Vanilla')
    products['Has_Organic'] = create_flag(products, 'Ingredients', 'Organic')
    products['Has_Caramel'] = create_flag(products, 'Ingredients', 'Caramel')
    products['Has_Chocolate'] = create_flag(products, 'Ingredients', 'Chocolat|Coco')
    products['Has_Fruit'] = create_flag(products, 'Ingredients', 'Raspberr|Cherr|Blueberr|Banana')

    is_bar_name = products['ProductName'].astype(str).str.contains('\bBar\b', case=False, regex=True)
    is_bar_sub = products['Subhead'].astype(str).str.contains('\bBar\b', case=False, regex=True)
    products['Is_Bar'] = (is_bar_name | is_bar_sub).astype(int)

    print("Data cleaning and preprocessing complete.")

    # ==============================================
    print("\n--- Generating Graphs and Visualizations ---")
    # ==============================================

    # 1. Overall Rating Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(products['Rating'], bins=10, kde=False, color='skyblue', edgecolor='blue')
    plt.title("Overall Ice Cream Product Rating Histogram")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    output_path = os.path.join(results_dir, "01_overall_rating_histogram.pdf")
    plt.savefig(output_path)
    plt.close()
    print("Saved: Overall Rating Histogram")

    # 2. QQ Plot
    plt.figure(figsize=(6, 6))
    stats.probplot(products['Rating'].dropna(), dist="norm", plot=plt)
    plt.title("QQ Plot of Product Ratings")
    output_path = os.path.join(results_dir, "02_rating_qq_plot.pdf")
    plt.savefig(output_path)
    plt.close()
    print("Saved: QQ Plot of Product Ratings")

    # 3. Boxplot by Brand
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=products, x='Brand', y='Rating', hue='Brand', legend=False, palette="Set3")
    plt.title("Distribution of Ratings Across Brands")
    output_path = os.path.join(results_dir, "03_comparative_brand_boxplot.pdf")
    plt.savefig(output_path)
    plt.close()
    print("Saved: Distribution of Ratings Across Brands")

    # 4. Density Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=products, x='Rating', hue='Brand', fill=True, alpha=0.4, palette="Set2")
    plt.title("Density of Ratings Across Brands")
    output_path = os.path.join(results_dir, "04_comparative_brand_density.pdf")
    plt.savefig(output_path)
    plt.close()
    print("Saved: Density of Ratings Across Brands")

    unique_brands = products['Brand'].unique()
    colors = {'bj': '#6dbf75', 'hd': '#dca963', 'talenti': '#a73e5c', 'breyers': '#337ab7'}

    for i, brand in enumerate(unique_brands):
        brand_data = products[products['Brand'] == brand].copy()
        color = colors.get(brand, 'gray') # Default to gray if brand not in dict
        
        # 5.x Brand Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(brand_data['Rating'], bins=10, color=color, edgecolor='black')
        plt.title(f"{brand.title()} - Product Rating Histogram")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        
        filename = f"05_{brand}_rating_histogram.pdf"
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path)
        plt.close()

        # 6.x Brand Scatter Plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=brand_data, x='ProductID', y='Rating', color=color, s=100)
        plt.xticks(rotation=90, fontsize=6)
        plt.title(f"{brand.title()} - Ratings per Product")
        plt.tight_layout()
        
        filename = f"06_{brand}_product_scatter.pdf"
        output_path = os.path.join(results_dir, filename)
        plt.savefig(output_path)
        plt.close()

    print("Saved: Individual brand graphs")

    # ==============================================
    print("\n--- Statistical Analysis ---")
    # ==============================================

    # Basic Stats
    mean_rating = products['Rating'].mean()
    sd_rating = products['Rating'].std()
    print(f"Overall Mean Rating: {mean_rating:.3f}")
    print(f"Overall Std Dev: {sd_rating:.3f}")

    # Shapiro-Wilk
    shapiro_stat, shapiro_p = stats.shapiro(products['Rating'].dropna())
    print(f"\nShapiro-Wilk P-Value: {shapiro_p:.5f} ({'Normal' if shapiro_p > 0.05 else 'Not Normal'})")

    # Comparison Stats Loop
    print("\nPairwise Comparisons (T-Tests):")
    brand_pairs = list(itertools.combinations(unique_brands, 2))

    for brand1, brand2 in brand_pairs:
        r1 = products[products['Brand'] == brand1]['Rating'].dropna()
        r2 = products[products['Brand'] == brand2]['Rating'].dropna()
        
        t_stat, t_p = stats.ttest_ind(r1, r2, equal_var=False)
        sig = "SIGNIFICANT" if t_p < 0.05 else "Not Significant"
        print(f"  {brand1} vs {brand2}: p-value = {t_p:.4f} ({sig})")

    # ==============================================
    print("\n--- Saving Processed Data ---")

    clean_products_path = os.path.join(cleaned_data_dir, "products_clean.csv")
    clean_reviews_path = os.path.join(cleaned_data_dir, "reviews_clean.csv")

    # Save to CSV without the pandas index column
    products.to_csv(clean_products_path, index=False)
    reviews.to_csv(clean_reviews_path, index=False)

    print("Cleaned data saved successfully.")

if __name__ == "__main__":
    run_eda()