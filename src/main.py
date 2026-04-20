import os
from data_collection import load_data
from data_cleaning import clean_data
from eda import generate_eda_stats
from visualization import create_visualizations
from feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def generate_html_report(stats, plots, model_scores, output_path='../outputs/report.html'):
    """
    Generate an HTML file displaying the EDA findings, plots, and a basic ML summary.
    """
    print(f"Generating HTML report at: {output_path}")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Pipeline Project - EDA Report</title>
        <!-- Bootstrap Core CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; padding-top: 2rem; background-color: #f8f9fa; }}
            .container {{ max-width: 1000px; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
            .section {{ margin-bottom: 3rem; margin-top: 3rem; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; margin-bottom: 15px; background: white; }}
            .table {{ font-size: 0.9rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center">Data Pipeline & EDA Report</h1>
            <p class="text-center text-muted">A full End-to-End Pipeline converting raw Titanic data into ML-ready formats.</p>

            <div class="section">
                <h2>1. Dataset Statistics</h2>
                <div class="row">
                    <div class="col-md-6">
                        <h4>Shape</h4>
                        <p>{stats['shape'][0]} Rows x {stats['shape'][1]} Columns</p>
                    </div>
                    <div class="col-md-6">
                        <h4>Missing Values Overview</h4>
                        {stats['missing_values']}
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>2. Statistical Description</h2>
                <div class="table-responsive">
                    {stats['describe']}
                </div>
            </div>

            <div class="section">
                <h2>3. Exploratory Data Analysis Plots</h2>
                <div class="row">
    """

    for plot_path in plots:
        # Convert path relative to the outputs folder
        rel_path = os.path.basename(plot_path)
        rel_path = f"plots/{rel_path}"
        html_content += f"""
                    <div class="col-md-6 text-center">
                        <img src="{rel_path}" alt="EDA Plot">
                    </div>
        """

    html_content += f"""
                </div>
            </div>

            <div class="section">
                <h2>4. Feature Correlation Matrix</h2>
                <div class="table-responsive">
                    {stats['correlation']}
                </div>
            </div>

            <div class="section">
                <h2>5. Machine Learning Summary (Bonus)</h2>
                <div class="alert alert-info">
                    <strong>Model:</strong> Random Forest Classifier<br>
                    <strong>Train Accuracy:</strong> {model_scores['train']:.2f}%<br>
                    <strong>Test Accuracy:</strong> {model_scores['test']:.2f}%
                </div>
                <h4>Final Summary of Insights</h4>
                <ul>
                    <li><strong>Demographics & Class:</strong> Lower class passengers had a substantially worse survival rate.</li>
                    <li><strong>Correlations:</strong> Age, Fare, and Pclass strongly correlate with survival.</li>
                    <li><strong>Preprocessing:</strong> Missing Age data was median-filled, Outlier Fares were capped, Categoricals mapped logically. Using 'FamilySize' helped clarify passenger groupings.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # Ensure output directory exists based on output_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Report HTML successfully built.\n")


def run_pipeline():
    # Setup working directory properly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'titanic.csv')
    plots_dir = os.path.join(base_dir, 'outputs', 'plots')
    report_path = os.path.join(base_dir, 'outputs', 'report.html')

    print("="*40)
    print("STARTING DATA PIPELINE")
    print("="*40)

    # 1. Load Data
    df = load_data(filepath=data_path)

    # 2. Clean Data
    df_cleaned = clean_data(df)

    # 3. EDA Statistics & Visualizations
    stats = generate_eda_stats(df_cleaned)
    plots_generated = create_visualizations(df_cleaned, output_dir=plots_dir)

    # 4. Feature Engineering
    df_engineered = feature_engineering(df_cleaned)

    # 5. ML Preprocessing (Bonus)
    X = df_engineered.drop('Survived', axis=1)
    y = df_engineered['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_scores = {
        'train': model.score(X_train, y_train) * 100,
        'test': model.score(X_test, y_test) * 100
    }

    # 6. Generate HTML Report
    generate_html_report(stats, plots_generated, model_scores, output_path=report_path)

    print("Pipeline Execution Completed Successfully.")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
