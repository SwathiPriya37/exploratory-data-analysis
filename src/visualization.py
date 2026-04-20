import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def create_visualizations(df, output_dir='../outputs/plots/'):
    """
    Generates 8 data visualization charts and saves them as images.
    """
    print(f"Creating and saving visualizations into: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plots_generated = []

    # 1. Histogram - Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True, bins=30, color='skyblue')
    plt.title('1. Histogram: Age Distribution')
    hist_path = os.path.join(output_dir, '1_histogram_age.png')
    plt.savefig(hist_path)
    plots_generated.append(hist_path)
    plt.close()

    # 2. Boxplot - Fare by Class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set2')
    plt.title('2. Boxplot: Fare by Passenger Class')
    box_path = os.path.join(output_dir, '2_boxplot_fare.png')
    plt.savefig(box_path)
    plots_generated.append(box_path)
    plt.close()

    # 3. Heatmap - Correlation Matrix
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('3. Heatmap: Correlation Matrix')
    heatmap_path = os.path.join(output_dir, '3_heatmap_corr.png')
    plt.savefig(heatmap_path)
    plots_generated.append(heatmap_path)
    plt.close()

    # 4. Countplot - Survival Count by Sex
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Survived', hue='Sex', data=df, palette='pastel')
    plt.title('4. Countplot: Survival by Sex')
    count_path = os.path.join(output_dir, '4_countplot_survival_sex.png')
    plt.savefig(count_path)
    plots_generated.append(count_path)
    plt.close()

    # 5. Scatter Plot - Age vs Fare
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='rainbow', alpha=0.7)
    plt.title('5. Scatter Plot: Age vs Fare')
    scatter_path = os.path.join(output_dir, '5_scatter_age_fare.png')
    plt.savefig(scatter_path)
    plots_generated.append(scatter_path)
    plt.close()

    # 6. Line Plot - Average Fare over Age groups
    plt.figure(figsize=(10, 6))
    age_fare = df.groupby(pd.cut(df['Age'], bins=10))['Fare'].mean().reset_index()
    age_fare['Age_mid'] = age_fare['Age'].apply(lambda x: x.mid)
    sns.lineplot(x='Age_mid', y='Fare', data=age_fare, marker='o', color='purple')
    plt.title('6. Line Plot: Avg Fare across Age Brackets')
    line_path = os.path.join(output_dir, '6_lineplot_avg_fare.png')
    plt.savefig(line_path)
    plots_generated.append(line_path)
    plt.close()

    # 7. Bar Chart - Survival Rate by Passenger Class
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis', errorbar=None)
    plt.title('7. Bar Chart: Survival Rate by Ticket Class')
    bar_path = os.path.join(output_dir, '7_barchart_class_survival.png')
    plt.savefig(bar_path)
    plots_generated.append(bar_path)
    plt.close()

    # 8. Pairplot - Numerical Features Relationships
    # Suppress warnings for pairplot KDEs
    import warnings
    warnings.filterwarnings('ignore')
    
    pairplot_cols = ['Survived', 'Pclass', 'Age', 'Fare', 'SibSp']
    # Filter columns to only include those present in df
    pairplot_cols = [c for c in pairplot_cols if c in df.columns]
    
    g = sns.pairplot(df[pairplot_cols], hue='Survived', palette='husl', corner=True)
    g.fig.suptitle('8. Pairplot: Numerical Features', y=1.02)
    pairplot_path = os.path.join(output_dir, '8_pairplot.png')
    plt.savefig(pairplot_path)
    plots_generated.append(pairplot_path)
    plt.close()

    print("Successfully saved 8 plots.\n")
    return plots_generated
