import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file_path):
    """
    Load CSV data and prepare it for advanced analysis.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing the dataset
        
    Returns:
    --------
    tuple
        (dataframe, available_framing_types, available_binary_columns)
    """
    df = pd.read_csv(csv_file_path)
    
    # Define framing types and binary columns
    framing_types = ['police_solidarity', 'protester_solidarity', 'peace', 'conflict']
    
    # Binary columns for analysis
    binary_columns = [
        'people_crowd_size_none', 'people_crowd_size_small_1_to_10', 
        'people_crowd_size_medium_11_to_50', 'people_crowd_size_large_50_plus', 
        'people_crowd_dynamics_chaotic_or_disorganized', 
        'people_crowd_dynamics_organized_or_unified', 
        'people_police_present', 'people_police_regular_gear', 
        'people_police_riot_gear', 'people_children_present',
        'actions_marching', 'actions_standing_confrontation', 
        'actions_action_fighting', 'actions_action_throwing_objects', 
        'actions_action_shouting', 'actions_damaging_property', 
        'actions_arrests_visible', 'actions_protester_violence', 
        'actions_police_using_force', 'actions_medical_aid', 
        'actions_speaking_or_chanting', 'actions_aggressive_gestures', 
        'actions_raised_fists_or_hands', 'actions_pointing_fingers', 
        'actions_kneeling', 'actions_linked_arms', 'actions_holding_hands', 
        'actions_peaceful_gathering', 'actions_vigils', 
        'actions_comforting_or_hugging', 'actions_retreating_or_running_from_police', 
        'actions_shields_raised', 'actions_firearms_raised', 
        'actions_distributing_supplies', 'actions_helping_injured', 
        'actions_batons_raised', 'actions_mutual_cover', 
        'actions_standing_in_line_or_wall_formation',
        'protester_emotions_happy', 'protester_emotions_angry', 
        'protester_emotions_somber', 'protester_emotions_determined', 
        'protester_emotions_fearful', 'protester_emotions_calm', 
        'protester_emotions_tense', 'police_emotions_happy', 
        'police_emotions_angry', 'police_emotions_somber', 
        'police_emotions_determined', 'police_emotions_fearful', 
        'police_emotions_calm', 'police_emotions_tense',
        'objects_weapons_visible', 'objects_projectiles', 
        'objects_barriers_or_fences', 'objects_shields', 
        'objects_signs_or_banners', 'objects_flags', 
        'objects_burning_or_trampled_flag', 'objects_megaphones_or_speakers', 
        'objects_cameras_or_phones', 'objects_injured_or_dead_bodies', 
        'objects_vehicle_none', 'objects_vehicle_civilian', 
        'objects_vehicle_police', 'objects_vehicle_emergency', 
        'objects_vehicle_unclear', 'objects_damage_visible', 
        'objects_graffiti', 'objects_smoke_or_fire', 
        'objects_debris_trash_or_garbage', 'objects_debris_glass_or_broken_glass', 
        'objects_memorial_elements', 'objects_umbrellas_or_improvised_shields',
        'environment_location_outdoor', 'environment_location_indoor', 
        'environment_location_unclear'
    ]
    
    # Filter columns that actually exist in the dataframe
    available_binary_columns = [col for col in binary_columns if col in df.columns]
    available_framing_types = [col for col in framing_types if col in df.columns]
    
    print(f"Found {len(available_binary_columns)} binary columns out of {len(binary_columns)} expected")
    print(f"Found {len(available_framing_types)} framing types out of {len(framing_types)} expected")
    print(f"Dataset shape: {df.shape}")
    
    # Convert to numeric and handle missing values
    for col in available_binary_columns + available_framing_types:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
    
    return df, available_framing_types, available_binary_columns

def bootstrap_logistic_regression(X, y, n_bootstrap=1000, random_state=42):
    """
    Perform bootstrap logistic regression to get confidence intervals for coefficients.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing bootstrap statistics and results
    """
    print(f"Performing bootstrap logistic regression with {n_bootstrap} iterations...")
    
    bootstrap_coefficients = []
    bootstrap_auc_scores = []
    
    np.random.seed(random_state)
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}")
        
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Fit logistic regression
        model = LogisticRegression(random_state=i, max_iter=1000, class_weight='balanced')
        model.fit(X_boot, y_boot)
        
        # Store coefficients
        bootstrap_coefficients.append(model.coef_[0])
        
        # Calculate AUC on bootstrap sample
        y_pred_proba = model.predict_proba(X_boot)[:, 1]
        auc_score = roc_auc_score(y_boot, y_pred_proba)
        bootstrap_auc_scores.append(auc_score)
    
    # Convert to numpy array for easier calculation
    bootstrap_coefficients = np.array(bootstrap_coefficients)
    bootstrap_auc_scores = np.array(bootstrap_auc_scores)
    
    # Calculate statistics
    mean_coefficients = np.mean(bootstrap_coefficients, axis=0)
    std_coefficients = np.std(bootstrap_coefficients, axis=0)
    ci_lower = np.percentile(bootstrap_coefficients, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_coefficients, 97.5, axis=0)
    
    mean_auc = np.mean(bootstrap_auc_scores)
    std_auc = np.std(bootstrap_auc_scores)
    ci_auc_lower = np.percentile(bootstrap_auc_scores, 2.5)
    ci_auc_upper = np.percentile(bootstrap_auc_scores, 97.5)
    
    return {
        'mean_coefficients': mean_coefficients,
        'std_coefficients': std_coefficients,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_auc_lower': ci_auc_lower,
        'ci_auc_upper': ci_auc_upper,
        'bootstrap_coefficients': bootstrap_coefficients,
        'bootstrap_auc_scores': bootstrap_auc_scores
    }

def bootstrap_random_forest(X, y, n_bootstrap=1000, random_state=42):
    """
    Perform bootstrap random forest to get confidence intervals for feature importance.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing bootstrap statistics and results
    """
    print(f"Performing bootstrap random forest with {n_bootstrap} iterations...")
    
    bootstrap_importances = []
    bootstrap_auc_scores = []
    
    np.random.seed(random_state)
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap iteration {i + 1}/{n_bootstrap}")
        
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Fit random forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=i, class_weight='balanced')
        rf_model.fit(X_boot, y_boot)
        
        # Store feature importances
        bootstrap_importances.append(rf_model.feature_importances_)
        
        # Calculate AUC on bootstrap sample
        y_pred_proba = rf_model.predict_proba(X_boot)[:, 1]
        auc_score = roc_auc_score(y_boot, y_pred_proba)
        bootstrap_auc_scores.append(auc_score)
    
    # Convert to numpy array for easier calculation
    bootstrap_importances = np.array(bootstrap_importances)
    bootstrap_auc_scores = np.array(bootstrap_auc_scores)
    
    # Calculate statistics
    mean_importances = np.mean(bootstrap_importances, axis=0)
    std_importances = np.std(bootstrap_importances, axis=0)
    ci_lower = np.percentile(bootstrap_importances, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_importances, 97.5, axis=0)
    
    mean_auc = np.mean(bootstrap_auc_scores)
    std_auc = np.std(bootstrap_auc_scores)
    ci_auc_lower = np.percentile(bootstrap_auc_scores, 2.5)
    ci_auc_upper = np.percentile(bootstrap_auc_scores, 97.5)
    
    return {
        'mean_importances': mean_importances,
        'std_importances': std_importances,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_auc_lower': ci_auc_lower,
        'ci_auc_upper': ci_auc_upper,
        'bootstrap_importances': bootstrap_importances,
        'bootstrap_auc_scores': bootstrap_auc_scores
    }

def logistic_regression_analysis_with_bootstrap(df, framing_types, binary_columns, n_bootstrap=1000):
    """
    Perform logistic regression analysis with bootstrap for each framing type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    framing_types : list
        List of framing type column names
    binary_columns : list
        List of binary feature column names
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
        
    Returns:
    --------
    dict
        Dictionary containing results for each framing type
    """
    results = {}
    
    print("Performing logistic regression analysis with bootstrap...")
    
    for framing in framing_types:
        print(f"\nAnalyzing {framing}...")
        
        # Prepare data
        X = df[binary_columns]
        y = df[framing]
        
        # Check if we have enough positive cases
        positive_cases = y.sum()
        if positive_cases < 10:
            print(f"  Warning: Only {positive_cases} positive cases for {framing}. Skipping logistic regression.")
            continue
        
        # Perform bootstrap analysis
        bootstrap_results = bootstrap_logistic_regression(X, y, n_bootstrap)
        
        # Create feature importance dataframe with confidence intervals
        feature_importance = pd.DataFrame({
            'feature': binary_columns,
            'coefficient': bootstrap_results['mean_coefficients'],
            'coefficient_std': bootstrap_results['std_coefficients'],
            'coefficient_ci_lower': bootstrap_results['ci_lower'],
            'coefficient_ci_upper': bootstrap_results['ci_upper'],
            'odds_ratio': np.exp(bootstrap_results['mean_coefficients']),
            'odds_ratio_ci_lower': np.exp(bootstrap_results['ci_lower']),
            'odds_ratio_ci_upper': np.exp(bootstrap_results['ci_upper']),
            'abs_coefficient': np.abs(bootstrap_results['mean_coefficients'])
        }).sort_values('abs_coefficient', ascending=False)
        
        # Calculate significance (CI doesn't include 0)
        feature_importance['coefficient_significant'] = (
            (feature_importance['coefficient_ci_lower'] > 0) | 
            (feature_importance['coefficient_ci_upper'] < 0)
        )
        
        results[framing] = {
            'feature_importance': feature_importance,
            'bootstrap_results': bootstrap_results,
            'mean_auc': bootstrap_results['mean_auc'],
            'auc_ci_lower': bootstrap_results['ci_auc_lower'],
            'auc_ci_upper': bootstrap_results['ci_auc_upper']
        }
        
        print(f"  Bootstrap AUC: {bootstrap_results['mean_auc']:.4f} (95% CI: {bootstrap_results['ci_auc_lower']:.4f}-{bootstrap_results['ci_auc_upper']:.4f})")
        print(f"  Top 5 features with 95% CI:")
        for _, row in feature_importance.head(5).iterrows():
            significance = "***" if row['coefficient_significant'] else ""
            print(f"    {row['feature']}: OR={row['odds_ratio']:.2f} (CI: {row['odds_ratio_ci_lower']:.2f}-{row['odds_ratio_ci_upper']:.2f}) {significance}")
    
    return results

def random_forest_analysis_with_bootstrap(df, framing_types, binary_columns, n_bootstrap=1000):
    """
    Perform random forest analysis with bootstrap for each framing type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    framing_types : list
        List of framing type column names
    binary_columns : list
        List of binary feature column names
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
        
    Returns:
    --------
    dict
        Dictionary containing results for each framing type
    """
    results = {}
    
    print("Performing random forest analysis with bootstrap...")
    
    for framing in framing_types:
        print(f"\nAnalyzing {framing} with Random Forest...")
        
        # Prepare data
        X = df[binary_columns]
        y = df[framing]
        
        # Check if we have enough positive cases
        positive_cases = y.sum()
        if positive_cases < 10:
            print(f"  Warning: Only {positive_cases} positive cases for {framing}. Skipping random forest.")
            continue
        
        # Perform bootstrap analysis
        bootstrap_results = bootstrap_random_forest(X, y, n_bootstrap)
        
        # Create feature importance dataframe with confidence intervals
        feature_importance = pd.DataFrame({
            'feature': binary_columns,
            'importance': bootstrap_results['mean_importances'],
            'importance_std': bootstrap_results['std_importances'],
            'importance_ci_lower': bootstrap_results['ci_lower'],
            'importance_ci_upper': bootstrap_results['ci_upper']
        }).sort_values('importance', ascending=False)
        
        # Calculate significance (CI doesn't include 0)
        feature_importance['importance_significant'] = feature_importance['importance_ci_lower'] > 0
        
        results[framing] = {
            'feature_importance': feature_importance,
            'bootstrap_results': bootstrap_results,
            'mean_auc': bootstrap_results['mean_auc'],
            'auc_ci_lower': bootstrap_results['ci_auc_lower'],
            'auc_ci_upper': bootstrap_results['ci_auc_upper']
        }
        
        print(f"  Bootstrap AUC: {bootstrap_results['mean_auc']:.4f} (95% CI: {bootstrap_results['ci_auc_lower']:.4f}-{bootstrap_results['ci_auc_upper']:.4f})")
        print(f"  Top 5 features with 95% CI:")
        for _, row in feature_importance.head(5).iterrows():
            significance = "***" if row['importance_significant'] else ""
            print(f"    {row['feature']}: Importance={row['importance']:.4f} (CI: {row['importance_ci_lower']:.4f}-{row['importance_ci_upper']:.4f}) {significance}")
    
    return results

def save_results(logistic_results, rf_results, output_dir):
    """
    Save all results to CSV files.
    
    Parameters:
    -----------
    logistic_results : dict
        Results from logistic regression analysis
    rf_results : dict
        Results from random forest analysis
    output_dir : str
        Directory to save output files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save logistic regression results
    for framing, results in logistic_results.items():
        results['feature_importance'].to_csv(f"{output_dir}/logistic_regression_bootstrap_{framing}.csv", index=False)
    
    # Save random forest results
    for framing, results in rf_results.items():
        results['feature_importance'].to_csv(f"{output_dir}/random_forest_bootstrap_{framing}.csv", index=False)
    
    print(f"Results saved to: {output_dir}")

def print_comprehensive_summary(logistic_results, rf_results):
    """
    Print comprehensive summary of all analyses.
    
    Parameters:
    -----------
    logistic_results : dict
        Results from logistic regression analysis
    rf_results : dict
        Results from random forest analysis
    """
    print("\n" + "="*80)
    print("BOOTSTRAP ANALYSIS SUMMARY")
    print("="*80)
    
    # Logistic regression summary
    print("\nLOGISTIC REGRESSION ANALYSIS (with Bootstrap):")
    for framing, results in logistic_results.items():
        print(f"  {framing}: AUC={results['mean_auc']:.4f} (95% CI: {results['auc_ci_lower']:.4f}-{results['auc_ci_upper']:.4f})")
    
    # Random forest summary
    print("\nRANDOM FOREST ANALYSIS (with Bootstrap):")
    for framing, results in rf_results.items():
        print(f"  {framing}: AUC={results['mean_auc']:.4f} (95% CI: {results['auc_ci_lower']:.4f}-{results['auc_ci_upper']:.4f})")

def main(csv_file_path, output_dir=None, n_bootstrap=1000):
    """
    Main function to execute bootstrap analysis.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the input CSV file
    output_dir : str, optional
        Directory to save output files
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
        
    Returns:
    --------
    tuple
        (logistic_results, rf_results)
    """
    print("Loading and preparing data...")
    df, framing_types, binary_columns = load_and_prepare_data(csv_file_path)
    
    print("Performing logistic regression analysis with bootstrap...")
    logistic_results = logistic_regression_analysis_with_bootstrap(df, framing_types, binary_columns, n_bootstrap)
    
    print("Performing random forest analysis with bootstrap...")
    rf_results = random_forest_analysis_with_bootstrap(df, framing_types, binary_columns, n_bootstrap)
    
    # Save results
    if output_dir:
        save_results(logistic_results, rf_results, output_dir)
    
    # Print comprehensive summary
    print_comprehensive_summary(logistic_results, rf_results)
    
    return logistic_results, rf_results

if __name__ == "__main__":
    # Example usage - modify these paths as needed
    csv_file_path = 'path/to/your/data.csv'
    output_dir = 'path/to/output/directory'
    
    try:
        logistic_results, rf_results = main(csv_file_path, output_dir, n_bootstrap=1000)
        print("\nBootstrap analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
