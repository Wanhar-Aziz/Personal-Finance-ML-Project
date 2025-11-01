import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from src.config import TRAIN_PATH, TARGET_CLASSIFICATION, RANDOM_STATE

def analyze_feature_leakage():
    """
    Analyze potential data leakage by examining feature importance and correlations
    """
    print("="*60)
    print("FEATURE LEAKAGE ANALYSIS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    train_df[TARGET_CLASSIFICATION] = train_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})
    
    # Features to analyze
    drop_features = ['user_id', 'record_date', 'has_loan', 'credit_score']
    known_leaky = ['loan_amount_usd', 'loan_term_months', 'monthly_emi_usd', 
                   'loan_interest_rate_pct', 'loan_type']
    
    # Separate numeric and categorical
    numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in drop_features + known_leaky]
    
    categorical_cols = ['gender', 'education_level', 'employment_status', 'job_title', 'region']
    
    # 1. Check correlations with numeric features
    print("\n1. NUMERIC FEATURE CORRELATIONS WITH TARGET:")
    print("-" * 60)
    correlations = []
    for col in numeric_cols:
        corr = train_df[col].corr(train_df[TARGET_CLASSIFICATION])
        correlations.append({'feature': col, 'correlation': abs(corr)})
        print(f"  {col:30s}: {corr:7.4f}")
    
    correlations_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    # 2. Check categorical feature predictive power
    print("\n2. CATEGORICAL FEATURE ANALYSIS:")
    print("-" * 60)
    
    for col in categorical_cols:
        # Calculate conditional probabilities
        contingency = pd.crosstab(train_df[col], train_df[TARGET_CLASSIFICATION], normalize='index')
        if 1 in contingency.columns:
            max_prob = contingency[1].max()
            min_prob = contingency[1].min()
            variance = contingency[1].var()
            print(f"\n  {col}:")
            print(f"    Max P(loan|{col}): {max_prob:.4f}")
            print(f"    Min P(loan|{col}): {min_prob:.4f}")
            print(f"    Variance: {variance:.4f}")
            if max_prob > 0.95 or min_prob < 0.05:
                print(f"    ⚠️  SUSPICIOUS: Near-deterministic relationship!")
    
    # 3. Train a decision tree and check feature importance
    print("\n3. DECISION TREE FEATURE IMPORTANCE:")
    print("-" * 60)
    
    X_train = train_df.drop(columns=drop_features + known_leaky)
    y_train = train_df[TARGET_CLASSIFICATION]
    
    # Encode categoricals
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_train[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    X_encoded = np.hstack([X_train[numeric_cols].values, X_cat_encoded])
    all_feature_names = numeric_cols + list(cat_feature_names)
    
    # Train tree
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
    tree.fit(X_encoded, y_train)
    
    # Get feature importance
    importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 15 Most Important Features:")
    for idx, row in importances.head(15).iterrows():
        print(f"    {row['feature']:40s}: {row['importance']:.4f}")
    
    # 4. Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Correlation with target
    ax1 = axes[0, 0]
    top_corr = correlations_df.head(10)
    ax1.barh(top_corr['feature'], top_corr['correlation'])
    ax1.set_xlabel('Absolute Correlation with has_loan')
    ax1.set_title('Top 10 Numeric Features by Correlation')
    ax1.invert_yaxis()
    
    # Plot 2: Feature importance
    ax2 = axes[0, 1]
    top_importance = importances.head(15)
    ax2.barh(top_importance['feature'], top_importance['importance'], color='coral')
    ax2.set_xlabel('Feature Importance')
    ax2.set_title('Top 15 Features by Decision Tree Importance')
    ax2.invert_yaxis()
    
    # Plot 3: Distribution of a suspicious numeric feature
    ax3 = axes[1, 0]
    suspicious_numeric = correlations_df.iloc[0]['feature']
    train_df.boxplot(column=suspicious_numeric, by=TARGET_CLASSIFICATION, ax=ax3)
    ax3.set_title(f'Distribution of {suspicious_numeric} by Loan Status')
    ax3.set_xlabel('has_loan')
    ax3.set_ylabel(suspicious_numeric)
    plt.sca(ax3)
    plt.xticks([1, 2], ['No', 'Yes'])
    
    # Plot 4: Categorical feature analysis
    ax4 = axes[1, 1]
    # Check employment_status as an example
    employment_loan = pd.crosstab(train_df['employment_status'], 
                                   train_df[TARGET_CLASSIFICATION], 
                                   normalize='index')
    if 1 in employment_loan.columns:
        employment_loan[1].plot(kind='bar', ax=ax4, color='steelblue')
    ax4.set_title('P(has_loan=Yes | employment_status)')
    ax4.set_ylabel('Probability')
    ax4.set_xlabel('Employment Status')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('leakage_analysis.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*60)
    print("✓ Visualization saved to 'leakage_analysis.png'")
    print("="*60)
    
    # 5. Check for missing value patterns
    print("\n4. MISSING VALUE PATTERNS:")
    print("-" * 60)
    for col in train_df.columns:
        if train_df[col].isna().sum() > 0:
            missing_pct = train_df[col].isna().mean() * 100
            # Check if missingness correlates with target
            has_loan_when_missing = train_df[train_df[col].isna()][TARGET_CLASSIFICATION].mean()
            has_loan_when_present = train_df[~train_df[col].isna()][TARGET_CLASSIFICATION].mean()
            diff = abs(has_loan_when_missing - has_loan_when_present)
            
            print(f"\n  {col}:")
            print(f"    Missing: {missing_pct:.2f}%")
            print(f"    P(loan|missing): {has_loan_when_missing:.4f}")
            print(f"    P(loan|present): {has_loan_when_present:.4f}")
            if diff > 0.3:
                print(f"    ⚠️  SUSPICIOUS: Missingness is highly predictive!")
    
    return importances

if __name__ == "__main__":
    importances = analyze_feature_leakage()