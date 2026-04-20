import pandas as pd

def generate_eda_stats(df):
    """
    Generates Exploratory Data Analysis statistics for reporting.
    """
    print("Generating EDA statistics...")

    stats = {
        'shape': df.shape,
        'info': df.info(buf=None), # info usually prints to console.
        'describe': df.describe().to_html(classes="table table-striped table-bordered"),
    }
    
    # Missing Values
    missing_vals = df.isnull().sum()
    stats['missing_values'] = missing_vals[missing_vals > 0].to_frame(name='Missing Count').to_html() if missing_vals.max() > 0 else "No missing values."

    # Correlation Matrix (Numerical features only)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr()
    
    # Formatting correlation matrix as HTML
    stats['correlation'] = corr_matrix.to_html(classes="table table-striped table-bordered", float_format=lambda x: f"{x:.2f}")

    print("EDA generation complete.\n")
    return stats
