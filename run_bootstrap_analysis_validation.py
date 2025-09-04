#!/usr/bin/env python3
"""
Bootstrap Analysis Script - Option A (High Consistency)
Implements Phase 2 of the Reliability and Robustness Analysis Roadmap
For models with high consistency (Krippendorff's α > 0.8)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis imports
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, cohen_kappa_score
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

BOOTSTRAP_N = 2000

class BootstrapAnalyzer:
    """
    Bootstrap analyzer for high-consistency models using combined run analysis.
    
    This class implements comprehensive bootstrap analysis for evaluating model performance
    across multiple runs and prompt levels, addressing four core goals:
    1. Quantify uncertainty in performance metrics
    2. Demonstrate robustness across runs and models
    3. Support claims of statistical significance
    4. Compare metrics across models with precision
    """
    
    def __init__(self, data_folder: str, gt_file: str, output_dir: Optional[str] = None):
        """
        Initialize the bootstrap analyzer.
        
        Args:
            data_folder: Path to folder containing all run data
            gt_file: Path to ground truth file
            output_dir: Directory to save outputs (auto-generated if None)
        """
        self.data_folder = data_folder
        self.gt_file = gt_file
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'bootstrap_analysis_{timestamp}'
        else:
            self.output_dir = output_dir
        
        # Configuration
        self.prompt_levels = [
            'detailed', 
            'expert', 
            'simple', 
            'v2',
            ]
        self.run_count = 3
        self.model_names = [
            'gemma3_27b_it_q8_0', 
            'gpt_4_1', 
            'internvl3_14b', 
            'internvl3_38b', 
            'qwen2_5',
            ]
        
        # Data containers
        self.bootstrap_results = {}
        self.framing_columns = ['police_solidarity', 'protester_solidarity', 'peace', 'conflict']
        
        # Load ground truth
        self.ground_truth = None
        self.binary_columns = []
        self.load_ground_truth()
    
    def get_binary_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of binary columns for analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that contain binary values
        """
        excluded_columns = [
            'filename',
            # Supporting elements (text fields, not binary predictions)
            'police_solidarity_supporting_elements',
            'protester_solidarity_supporting_elements', 
            'peace_supporting_elements',
            'conflict_supporting_elements',
            # Non-prediction columns
            'model_used', 'confidence_calibration', 'ensemble_strategy',
            'ensemble_num_models', 'ensemble_models', 'reasoning_chain',
            'raw_llm_output', 'people_number_of_people'
        ]
        
        binary_columns = []
        for col in df.columns:
            if col not in excluded_columns:
                # Check if column contains only True/False values (with possible NaN)
                unique_values = pd.Series(df[col]).dropna().unique()
                if len(unique_values) > 0 and set(unique_values).issubset({True, False, 0, 1}):
                    binary_columns.append(col)
        
        return binary_columns
    
    def load_ground_truth(self):
        """Load and prepare ground truth data."""
        try:
            self.ground_truth = pd.read_csv(self.gt_file).set_index('filename')
            self.binary_columns = self.get_binary_columns(self.ground_truth)
            print(f"Loaded ground truth: {len(self.ground_truth)} samples, {len(self.binary_columns)} binary columns")
        except Exception as e:
            raise ValueError(f"Error loading ground truth file: {e}")
    
    def majority_vote_combination(self, run1: pd.DataFrame, run2: pd.DataFrame, run3: pd.DataFrame) -> pd.DataFrame:
        """
        Combine three boolean prediction DataFrames using majority voting.
        
        Args:
            run1, run2, run3: DataFrames with binary predictions
        
        Returns:
            DataFrame with majority-voted predictions
        """
        # Align indices
        common_index = run1.index.intersection(run2.index).intersection(run3.index)
        common_columns = run1.columns.intersection(run2.columns).intersection(run3.columns)
        
        # Get aligned data
        run1_aligned = run1.loc[common_index, common_columns].fillna(False).astype(bool)
        run2_aligned = run2.loc[common_index, common_columns].fillna(False).astype(bool)
        run3_aligned = run3.loc[common_index, common_columns].fillna(False).astype(bool)
        
        # Stack predictions: shape (3, n_images, n_labels)
        all_runs = np.stack([run1_aligned.values, run2_aligned.values, run3_aligned.values])
        
        # Sum across runs (True=1, False=0)
        vote_sums = np.sum(all_runs, axis=0)
        
        # Majority vote: if sum >= 2, then True
        majority_predictions = vote_sums >= 2
        
        # Convert back to DataFrame
        result_df = pd.DataFrame(
            majority_predictions, 
            index=common_index, 
            columns=common_columns
        )
        
        return result_df
    
    def load_model_runs(self, model: str, prompt: str) -> Optional[pd.DataFrame]:
        """
        Load and combine three runs for a model-prompt combination.
        
        Args:
            model: Model name
            prompt: Prompt level
            
        Returns:
            Combined predictions using majority voting, or None if insufficient runs
        """
        runs = []
        
        for run_id in range(self.run_count):
            directory = os.path.join(self.data_folder, f'run{run_id}_{model}_{prompt}')
            
            if not os.path.exists(directory):
                continue
            
            # Find CSV files
            csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
            if not csv_files:
                continue
            
            try:
                # Load the first CSV file
                run_data = pd.read_csv(os.path.join(directory, csv_files[0]))
                if 'filename' in run_data.columns:
                    run_data = run_data.set_index('filename')
                    
                    # Only keep binary columns that exist in ground truth
                    available_columns = [col for col in self.binary_columns if col in run_data.columns]
                    if available_columns:
                        runs.append(run_data[available_columns])
                    
            except Exception as e:
                print(f"Warning: Error loading {directory}/{csv_files[0]}: {e}")
                continue
        
        if len(runs) >= 2:  # Need at least 2 runs
            if len(runs) == 3:
                # Three runs - use majority voting
                return self.majority_vote_combination(runs[0], runs[1], runs[2])
            else:  # Two runs - average and threshold at 0.5
                common_index = runs[0].index.intersection(runs[1].index)
                common_columns = set(runs[0].columns).intersection(set(runs[1].columns))
                common_columns = [col for col in self.binary_columns if col in common_columns]
                
                run1_aligned = runs[0].loc[common_index, common_columns].fillna(False).astype(float)
                run2_aligned = runs[1].loc[common_index, common_columns].fillna(False).astype(float)
                
                average_pred = (run1_aligned + run2_aligned) / 2
                return (average_pred >= 0.5).astype(bool)
        
        return None
    
    def calculate_single_metrics(self, y_true, y_pred):
        """
        Calculate all metrics for a single bootstrap sample.
        
        Args:
            y_true: Ground truth binary labels (numpy array)
            y_pred: Predicted binary labels (numpy array)
        
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        try:
            # Convert to integers for calculation
            y_true_int = y_true.astype(int)
            y_pred_int = y_pred.astype(int)
            
            # Check for variation
            unique_true = len(np.unique(y_true_int))
            unique_pred = len(np.unique(y_pred_int))
            
            # Pearson correlation
            if unique_true > 1 and unique_pred > 1:
                corr, _ = pearsonr(y_true_int, y_pred_int)
                metrics['pearson_correlation'] = corr if not np.isnan(corr) else 0.0
            else:
                metrics['pearson_correlation'] = 0.0
            
            # Matthews Correlation Coefficient
            try:
                mcc = matthews_corrcoef(y_true_int, y_pred_int)
                metrics['mcc'] = mcc if not np.isnan(mcc) else 0.0
            except:
                metrics['mcc'] = 0.0
            
            # F1-score
            try:
                f1 = f1_score(y_true_int, y_pred_int, zero_division=0)
                metrics['f1_score'] = f1 if not np.isnan(f1) else 0.0
            except:
                metrics['f1_score'] = 0.0
            
            # Balanced Accuracy
            try:
                balanced_acc = balanced_accuracy_score(y_true_int, y_pred_int)
                metrics['balanced_accuracy'] = balanced_acc if not np.isnan(balanced_acc) else 0.5
            except:
                metrics['balanced_accuracy'] = 0.5
            
            # Cohen's Kappa
            try:
                kappa = cohen_kappa_score(y_true_int, y_pred_int)
                metrics['cohens_kappa'] = kappa if not np.isnan(kappa) else 0.0
            except:
                metrics['cohens_kappa'] = 0.0
                
        except Exception as e:
            # If any error occurs, return zeros/baselines
            metrics = {
                'pearson_correlation': 0.0,
                'mcc': 0.0,
                'f1_score': 0.0,
                'balanced_accuracy': 0.5,
                'cohens_kappa': 0.0
            }
        
        return metrics
    
    def interpret_metric_significance(self, metric_name: str, ci_lower: float, ci_upper: float) -> bool:
        """
        Determine statistical significance based on metric type and confidence interval.
        
        Args:
            metric_name: Name of the metric
            ci_lower: Lower bound of confidence interval
            ci_upper: Upper bound of confidence interval
            
        Returns:
            True if statistically significant, False otherwise
        """
        if np.isnan(ci_lower) or np.isnan(ci_upper):
            return False
        
        # Define baselines for different metrics
        baselines = {
            'pearson_correlation': 0.0,
            'mcc': 0.0,
            'f1_score': 0.0,
            'balanced_accuracy': 0.5,  # Random performance baseline
            'cohens_kappa': 0.0
        }
        
        baseline = baselines.get(metric_name, 0.0)
        
        # Check if CI excludes the baseline
        return not (ci_lower <= baseline <= ci_upper)
    
    def interpret_effect_size(self, metric_name: str, value: float) -> str:
        """
        Interpret effect size based on metric type and value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            String describing the effect size
        """
        if np.isnan(value):
            return "Unable to calculate"
        
        abs_value = abs(value)
        
        if metric_name in ['pearson_correlation', 'mcc']:
            # Correlation-based metrics
            if abs_value > 0.7:
                return "Very Large"
            elif abs_value > 0.5:
                return "Large"
            elif abs_value > 0.3:
                return "Medium"
            elif abs_value > 0.1:
                return "Small"
            else:
                return "Negligible"
        
        elif metric_name == 'f1_score':
            # F1-score interpretation
            if value > 0.9:
                return "Excellent"
            elif value > 0.8:
                return "Very Good"
            elif value > 0.6:
                return "Good"
            elif value > 0.4:
                return "Fair"
            else:
                return "Poor"
        
        elif metric_name == 'balanced_accuracy':
            # Balanced accuracy interpretation
            if value > 0.95:
                return "Excellent"
            elif value > 0.9:
                return "Very Good"
            elif value > 0.8:
                return "Good"
            elif value > 0.7:
                return "Fair"
            else:
                return "Poor"
        
        elif metric_name == 'cohens_kappa':
            # Kappa interpretation (Landis & Koch)
            if abs_value > 0.8:
                return "Excellent"
            elif abs_value > 0.6:
                return "Good"
            elif abs_value > 0.4:
                return "Moderate"
            elif abs_value > 0.2:
                return "Fair"
            else:
                return "Poor"
        
        return "Unknown"

    def calculate_p_value_from_bootstrap(self, bootstrap_values, null_hypothesis=0.0):
        """
        Calculate p-value from bootstrap distribution.
        
        Args:
            bootstrap_values: Array of bootstrap statistics
            null_hypothesis: Value under null hypothesis (0 for correlations, 0.5 for balanced accuracy)
        
        Returns:
            Two-tailed p-value
        """
        if len(bootstrap_values) == 0:
            return 1.0
        
        # For two-tailed test
        if null_hypothesis == 0:
            # Proportion of bootstrap values as extreme as observed
            p_value = np.mean(np.abs(bootstrap_values) >= np.abs(np.mean(bootstrap_values)))
        else:
            # For metrics like balanced accuracy where null is 0.5
            observed_diff = np.mean(bootstrap_values) - null_hypothesis
            p_value = np.mean(np.abs(bootstrap_values - null_hypothesis) >= np.abs(observed_diff))
        
        return max(p_value, 1/len(bootstrap_values))  # Minimum p-value is 1/n_bootstrap

    def bootstrap_single_metrics(self, ground_truth_series: pd.Series, 
                                prediction_series: pd.Series, 
                                n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Perform bootstrap analysis for multiple metrics on a single label.
        
        Args:
            ground_truth_series: Ground truth labels
            prediction_series: Model predictions
            n_bootstrap: Number of bootstrap iterations
        
        Returns:
            Dictionary containing statistics for each metric
        """
        # Align indices
        common_index = ground_truth_series.index.intersection(prediction_series.index)
        gt_aligned = ground_truth_series[common_index].fillna(False).astype(bool)
        pred_aligned = prediction_series[common_index].fillna(False).astype(bool)
        
        if len(common_index) < 10:
            # Return empty results for insufficient data
            empty_result = {
                'sample_size': len(common_index),
                'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'median': np.nan, 'statistically_significant': False, 'effect_size': 'Unable to calculate',
                'precision': np.nan, 'values': []
            }
            return {
                'pearson_correlation': empty_result.copy(),
                'mcc': empty_result.copy(),
                'f1_score': empty_result.copy(),
                'balanced_accuracy': empty_result.copy(),
                'cohens_kappa': empty_result.copy()
            }
        
        # Initialize storage for bootstrap results
        bootstrap_metrics = {
            'pearson_correlation': [],
            'mcc': [],
            'f1_score': [],
            'balanced_accuracy': [],
            'cohens_kappa': []
        }
        
        # Perform bootstrap iterations
        for i in range(n_bootstrap):
            # Bootstrap sample with replacement
            n_samples = len(gt_aligned)
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Get bootstrapped samples
            boot_gt = gt_aligned.iloc[boot_indices].values
            boot_pred = pred_aligned.iloc[boot_indices].values
            
            # Calculate all metrics for this bootstrap sample
            sample_metrics = self.calculate_single_metrics(boot_gt, boot_pred)
            
            # Store results
            for metric_name, value in sample_metrics.items():
                bootstrap_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        results = {}
        
        for metric_name, values in bootstrap_metrics.items():
            values = np.array(values)
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                median_val = np.median(values)
                null_value = 0.5 if metric_name == 'balanced_accuracy' else 0.0
                p_value = self.calculate_p_value_from_bootstrap(values, null_value)

                # Statistical significance
                is_significant = self.interpret_metric_significance(metric_name, ci_lower, ci_upper)
                
                # Effect size
                effect_size = self.interpret_effect_size(metric_name, mean_val)
                
                # Precision (CI width)
                precision = ci_upper - ci_lower if not np.isnan(ci_upper) and not np.isnan(ci_lower) else np.nan
                
                results[metric_name] = {
                    'sample_size': len(common_index),
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'median': median_val,
                    'p_value': p_value,
                    'statistically_significant': is_significant,
                    'effect_size': effect_size,
                    'precision': precision,
                    'values': values.tolist()
                }
            else:
                # Empty results
                results[metric_name] = {
                    'sample_size': len(common_index),
                    'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                    'median': np.nan, 'statistically_significant': False, 
                    'effect_size': 'Unable to calculate', 'precision': np.nan, 'values': []
                }
        
        return results
    
    def analyze_model_prompt(self, model: str, prompt: str, n_bootstrap: int = 1000) -> Dict[str, Dict]:
        """
        Perform bootstrap analysis for a specific model-prompt combination.
        
        Args:
            model: Model name
            prompt: Prompt level
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Results organized by column, then by metric
        """
        print(f"Bootstrapping {model} - {prompt}")
        
        # Load and combine runs
        combined_predictions = self.load_model_runs(model, prompt)
        
        if combined_predictions is None:
            print(f"  Warning: Insufficient runs for {model}-{prompt}")
            return {}
        
        results = {}
        
        # Analyze each binary column
        available_columns = [col for col in self.binary_columns 
                           if col in combined_predictions.columns and col in self.ground_truth.columns]
        
        print(f"  Analyzing {len(available_columns)} columns...")
        
        for col in available_columns:
            gt_series = self.ground_truth[col]
            pred_series = combined_predictions[col]
            
            # Get bootstrap results for all metrics
            bootstrap_results = self.bootstrap_single_metrics(
                gt_series, pred_series, n_bootstrap
            )
            
            results[col] = bootstrap_results
        
        # Print summary for this model-prompt combination
        total_analyses = len(available_columns) * 5  # 5 metrics per column
        significant_count = 0
        for col_results in results.values():
            for metric_results in col_results.values():
                if metric_results['statistically_significant']:
                    significant_count += 1
        
        print(f"  Completed: {significant_count}/{total_analyses} statistically significant results")
        
        return results
    
    def run_full_bootstrap_analysis(self, n_bootstrap: int = 1000):
        """
        Run bootstrap analysis for all model-prompt combinations.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary containing all bootstrap results
        """
        print("Starting Bootstrap Analysis (Option A: High Consistency)")
        print(f"Models: {self.model_names}")
        print(f"Prompt levels: {self.prompt_levels}")
        print(f"Bootstrap iterations: {n_bootstrap}")
        print(f"Ground truth samples: {len(self.ground_truth)}")
        
        for model in self.model_names:
            self.bootstrap_results[model] = {}
            
            for prompt in self.prompt_levels:
                model_prompt_results = self.analyze_model_prompt(model, prompt, n_bootstrap)
                
                if model_prompt_results:
                    self.bootstrap_results[model][prompt] = model_prompt_results
                else:
                    print(f"  Skipping {model}-{prompt}: No valid results")
        
        print("\nBootstrap analysis complete!")
        return self.bootstrap_results

    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate comprehensive summary report of bootstrap results.
        
        Each row represents one metric for one model-prompt-category combination.
        
        Returns:
            DataFrame containing the summary report
        """
        report_data = []
        
        for model, prompt_results in self.bootstrap_results.items():
            for prompt, col_results in prompt_results.items():
                for col, metric_results in col_results.items():
                    for metric_name, bootstrap_result in metric_results.items():
                        
                        report_data.append({
                            'model': model,
                            'prompt': prompt,
                            'category': col,
                            'metric': metric_name,
                            'is_framing_column': col in self.framing_columns,
                            'sample_size': bootstrap_result.get('sample_size', 0),
                            'mean': bootstrap_result.get('mean', np.nan),
                            'std': bootstrap_result.get('std', np.nan),
                            'median': bootstrap_result.get('median', np.nan),
                            'ci_lower': bootstrap_result.get('ci_lower', np.nan),
                            'ci_upper': bootstrap_result.get('ci_upper', np.nan),
                            'p_value': bootstrap_result.get('p_value', np.nan),
                            'statistically_significant': bootstrap_result.get('statistically_significant', False),
                            'effect_size': bootstrap_result.get('effect_size', 'Unable to calculate'),
                            'precision': bootstrap_result.get('precision', np.nan)
                        })
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Apply FDR correction - for each metric separately
        if len(report_df) > 0 and 'p_value' in report_df.columns:
            # Group by metric and apply FDR correction within each metric type
            corrected_dfs = []
            for metric in report_df['metric'].unique():
                metric_df = report_df[report_df['metric'] == metric].copy()
                metric_df = self.apply_fdr_correction(metric_df, alpha=0.05)
                corrected_dfs.append(metric_df)
            
            report_df = pd.concat(corrected_dfs, ignore_index=True)

        return report_df
    
    def apply_fdr_correction(self, df, alpha=0.05):
        """
        Apply False Discovery Rate correction for multiple testing.
        
        Args:
            df: DataFrame with p_value column
            alpha: Significance level (default 0.05)
        
        Returns:
            DataFrame with additional corrected_significant column
        """
        from statsmodels.stats.multitest import multipletests
        
        # If no p_value column exists, we can't do FDR correction
        # In this case, we'll use the original statistically_significant values
        if 'p_value' not in df.columns:
            print("Note: No p_value column found, using original significance values for FDR correction")
            # Create a dummy p_value column based on statistical significance
            # For statistically significant results, we'll use a small p-value (0.01)
            # For non-significant results, we'll use a large p-value (0.5)
            df['p_value'] = np.where(df['statistically_significant'], 0.01, 0.5)
        
        # Remove NaN p-values for correction
        valid_mask = ~df['p_value'].isna()
        valid_p_values = df.loc[valid_mask, 'p_value'].values
        
        if len(valid_p_values) > 0:
            # Apply FDR correction
            rejected, corrected_p, alpha_sidak, alpha_bonf = multipletests(
                valid_p_values, 
                method='fdr_bh',  # Benjamini-Hochberg FDR
                alpha=alpha
            )
            
            # Add results back to dataframe
            df.loc[valid_mask, 'corrected_p_value'] = corrected_p
            df.loc[valid_mask, 'corrected_significant'] = rejected
            df.loc[~valid_mask, 'corrected_p_value'] = np.nan
            df.loc[~valid_mask, 'corrected_significant'] = False
            
            # Add correction method info
            df['correction_method'] = 'fdr_bh'
            df['correction_alpha'] = alpha
            
            # Log the correction impact
            original_significant = df['statistically_significant'].sum()
            corrected_significant = df['corrected_significant'].sum()
            print(f"  FDR Correction Impact: {original_significant} → {corrected_significant} significant results")
        else:
            df['corrected_significant'] = False
            df['corrected_p_value'] = np.nan
        
        return df

    def calculate_model_difference_significance(self, model1_results: Dict, model2_results: Dict, 
                                             metric: str, n_bootstrap: int = 1000) -> Dict:
        """
        Calculate statistical significance of differences between two models for a specific metric.
        
        Args:
            model1_results: Bootstrap results for first model
            model2_results: Bootstrap results for second model  
            metric: Metric name to compare
            n_bootstrap: Number of bootstrap iterations for difference calculation
        
        Returns:
            Dictionary with difference statistics and significance
        """
        # Get bootstrap values for both models
        if metric not in model1_results or metric not in model2_results:
            return None
            
        values1 = np.array(model1_results[metric]['values'])
        values2 = np.array(model2_results[metric]['values'])
        
        if len(values1) == 0 or len(values2) == 0:
            return None
        
        # Calculate observed difference
        observed_diff = model1_results[metric]['mean'] - model2_results[metric]['mean']
        
        # Bootstrap the difference
        n_samples = min(len(values1), len(values2))
        diff_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement from both distributions
            indices1 = np.random.choice(n_samples, size=n_samples, replace=True)
            indices2 = np.random.choice(n_samples, size=n_samples, replace=True)
            
            sample_diff = np.mean(values1[indices1]) - np.mean(values2[indices2])
            diff_bootstrap.append(sample_diff)
        
        diff_bootstrap = np.array(diff_bootstrap)
        
        # Calculate confidence interval for difference
        ci_lower = np.percentile(diff_bootstrap, 2.5)
        ci_upper = np.percentile(diff_bootstrap, 97.5)
        
        # Test if difference is significantly different from zero
        # Two-tailed test: check if CI excludes zero
        is_significant = not (ci_lower <= 0 <= ci_upper)
        
        # Calculate p-value for difference
        # Calculate p-value: proportion of bootstrap differences as extreme as observed
        if observed_diff >= 0:
            # For positive observed difference, count how many bootstrap differences are >= observed_diff
            p_value = np.mean(diff_bootstrap >= observed_diff)
        else:
            # For negative observed difference, count how many bootstrap differences are <= observed_diff
            p_value = np.mean(diff_bootstrap <= observed_diff)
        
        # Make it two-tailed
        p_value = min(p_value * 2, 1.0)
        
        # Ensure minimum p-value
        p_value = max(p_value, 1/len(diff_bootstrap))
        
        # Debug: Show p-value calculation details for first few comparisons
        if len(diff_bootstrap) > 0 and np.random.random() < 0.01:  # Show ~1% of calculations
            print(f"      Debug p-value calc: obs_diff={observed_diff:.4f}, p_value={p_value:.6f}, "
                  f"bootstrap_range=[{diff_bootstrap.min():.4f}, {diff_bootstrap.max():.4f}]")
        
        # Effect size interpretation for differences
        abs_diff = abs(observed_diff)
        if metric in ['pearson_correlation', 'mcc', 'cohens_kappa']:
            if abs_diff > 0.2:
                effect_size = "Large"
            elif abs_diff > 0.1:
                effect_size = "Medium" 
            elif abs_diff > 0.05:
                effect_size = "Small"
            else:
                effect_size = "Negligible"
        elif metric == 'balanced_accuracy':
            if abs_diff > 0.15:
                effect_size = "Large"
            elif abs_diff > 0.1:
                effect_size = "Medium"
            elif abs_diff > 0.05:
                effect_size = "Small"
            else:
                effect_size = "Negligible"
        elif metric == 'f1_score':
            if abs_diff > 0.2:
                effect_size = "Large"
            elif abs_diff > 0.1:
                effect_size = "Medium"
            elif abs_diff > 0.05:
                effect_size = "Small"
            else:
                effect_size = "Negligible"
        else:
            effect_size = "Unknown"
        
        return {
            'observed_difference': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'statistically_significant': is_significant,
            'effect_size': effect_size,
            'bootstrap_values': diff_bootstrap.tolist()
        }

    def analyze_cross_run_stability(self) -> Dict:
        """
        Analyze stability of results across different runs to demonstrate robustness.
        
        This addresses Goal 2: Demonstrate Robustness Across Runs and Models.
        
        Returns:
            Dictionary containing stability metrics for each model-prompt combination
        """
        print("Analyzing cross-run stability...")
        
        stability_results = {}
        
        for model in self.model_names:
            stability_results[model] = {}
            
            for prompt in self.prompt_levels:
                # Load individual runs
                runs = []
                for run_id in range(self.run_count):
                    directory = os.path.join(self.data_folder, f'run{run_id}_{model}_{prompt}')
                    if os.path.exists(directory):
                        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                        if csv_files:
                            try:
                                run_data = pd.read_csv(os.path.join(directory, csv_files[0]))
                                if 'filename' in run_data.columns:
                                    run_data = run_data.set_index('filename')
                                    available_columns = [col for col in self.binary_columns if col in run_data.columns]
                                    if available_columns:
                                        runs.append(run_data[available_columns])
                            except Exception as e:
                                continue
                
                if len(runs) >= 2:
                    # Calculate stability metrics for each column
                    stability_metrics = {}
                    
                    for col in self.binary_columns:
                        if col in runs[0].columns:
                            # Get predictions for this column across runs
                            col_predictions = []
                            for run in runs:
                                if col in run.columns:
                                    col_pred = run[col].fillna(False).astype(bool)
                                    col_predictions.append(col_pred)
                            
                            if len(col_predictions) >= 2:
                                # Calculate agreement between runs
                                agreements = []
                                for i in range(len(col_predictions)):
                                    for j in range(i+1, len(col_predictions)):
                                        # Align indices
                                        common_idx = col_predictions[i].index.intersection(col_predictions[j].index)
                                        if len(common_idx) > 0:
                                            pred1 = col_predictions[i].loc[common_idx]
                                            pred2 = col_predictions[j].loc[common_idx]
                                            agreement = (pred1 == pred2).mean()
                                            agreements.append(agreement)
                                
                                if agreements:
                                    stability_metrics[col] = {
                                        'mean_agreement': np.mean(agreements),
                                        'agreement_std': np.std(agreements),
                                        'min_agreement': np.min(agreements),
                                        'max_agreement': np.max(agreements),
                                        'n_run_pairs': len(agreements)
                                    }
                    
                    stability_results[model][prompt] = stability_metrics
        
        return stability_results

    def compare_models_statistically(self, metric: str) -> pd.DataFrame:
        """
        Perform statistical comparison of all model pairs for a specific metric.
        
        This addresses Goal 4: Compare Metrics Across Models with Precision.
        
        Args:
            metric: Metric name to compare across models
            
        Returns:
            DataFrame containing model comparison results
        """
        print(f"Performing statistical model comparisons for {metric}...")
        
        comparison_data = []
        
        # Get all model-prompt combinations that have results for this metric
        valid_combinations = []
        for model, prompt_results in self.bootstrap_results.items():
            for prompt, col_results in prompt_results.items():
                for col, metric_results in col_results.items():
                    if metric in metric_results and 'values' in metric_results[metric]:
                        valid_combinations.append((model, prompt, col, metric_results[metric]))
        
        print(f"  Found {len(valid_combinations)} valid combinations for {metric}")
        
        # Compare all pairs
        for i, (model1, prompt1, col1, results1) in enumerate(valid_combinations):
            for j, (model2, prompt2, col2, results2) in enumerate(valid_combinations[i+1:], i+1):
                
                # Only compare same category
                if col1 != col2:
                    continue
                
                # Calculate difference significance
                diff_stats = self.calculate_model_difference_significance(
                    {metric: results1}, {metric: results2}, metric
                )
                
                if diff_stats:
                    comparison_data.append({
                        'model1': model1,
                        'prompt1': prompt1,
                        'model2': model2,
                        'prompt2': prompt2,
                        'category': col1,
                        'metric': metric,
                        'model1_mean': results1['mean'],
                        'model2_mean': results2['mean'],
                        'difference': diff_stats['observed_difference'],
                        'difference_ci_lower': diff_stats['ci_lower'],
                        'difference_ci_upper': diff_stats['ci_upper'],
                        'difference_p_value': diff_stats['p_value'],
                        'difference_significant': diff_stats['statistically_significant'],
                        'difference_effect_size': diff_stats['effect_size']
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Debug: Show some statistics before FDR correction
            print(f"    Before FDR correction: {comparison_df['difference_significant'].sum()}/{len(comparison_df)} significant differences")
            print(f"    P-value range: {comparison_df['difference_p_value'].min():.6f} to {comparison_df['difference_p_value'].max():.6f}")
            
            # Apply FDR correction to difference p-values
            if len(comparison_df) > 1:
                comparison_df = self.apply_fdr_correction_to_comparisons(comparison_df)
            
            return comparison_df
        else:
            return pd.DataFrame()

    def apply_fdr_correction_to_comparisons(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply FDR correction to model comparison p-values.
        
        Args:
            comparison_df: DataFrame containing model comparison results
            
        Returns:
            DataFrame with FDR correction applied
        """
        if 'difference_p_value' not in comparison_df.columns:
            print("    Warning: No difference_p_value column found for FDR correction")
            return comparison_df
        
        # Remove NaN p-values for correction
        valid_mask = ~comparison_df['difference_p_value'].isna()
        valid_p_values = comparison_df.loc[valid_mask, 'difference_p_value'].values
        
        print(f"    FDR correction: {len(valid_p_values)} valid p-values out of {len(comparison_df)} total")
        
        if len(valid_p_values) > 0:
            # Debug: Show p-value distribution
            print(f"    P-value distribution: min={valid_p_values.min():.6f}, max={valid_p_values.max():.6f}")
            print(f"    P-values < 0.001: {(valid_p_values < 0.001).sum()}")
            print(f"    P-values < 0.01: {(valid_p_values < 0.01).sum()}")
            print(f"    P-values < 0.05: {(valid_p_values < 0.05).sum()}")
            
            # Apply FDR correction
            rejected, corrected_p, alpha_sidak, alpha_bonf = multipletests(
                valid_p_values, 
                method='fdr_bh',
                alpha=0.05
            )
            
            # Add results back to dataframe
            comparison_df.loc[valid_mask, 'corrected_difference_p_value'] = corrected_p
            comparison_df.loc[valid_mask, 'corrected_difference_significant'] = rejected
            comparison_df.loc[~valid_mask, 'corrected_difference_p_value'] = np.nan
            comparison_df.loc[~valid_mask, 'corrected_difference_significant'] = False
            
            # Log the correction impact
            original_significant = comparison_df['difference_significant'].sum()
            corrected_significant = comparison_df['corrected_difference_significant'].sum()
            print(f"    Model Comparison FDR Correction: {original_significant} → {corrected_significant} significant differences")
            
            # Debug: Show corrected p-value distribution
            if corrected_significant > 0:
                corrected_p_values = comparison_df.loc[comparison_df['corrected_difference_significant'], 'corrected_difference_p_value']
                print(f"    Corrected p-values for significant results: min={corrected_p_values.min():.6f}, max={corrected_p_values.max():.6f}")
        
        return comparison_df


    def save_results(self):
        """Save all bootstrap results and generate reports."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate summary report
        report_df = self.generate_summary_report()
        
        # Save detailed CSV report
        report_df.to_csv(os.path.join(self.output_dir, 'bootstrap_detailed_results.csv'), index=False)
        
        # Save JSON results (without full correlation arrays to reduce size)
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Create summary version without correlation arrays
        summary_results = {}
        for model, prompt_results in self.bootstrap_results.items():
            summary_results[model] = {}
            for prompt, col_results in prompt_results.items():
                summary_results[model][prompt] = {}
                for col, result in col_results.items():
                    summary_result = {k: v for k, v in result.items() if k != 'correlations'}
                    summary_results[model][prompt][col] = summary_result
        
        summary_results = convert_numpy(summary_results)
        with open(os.path.join(self.output_dir, 'bootstrap_summary_results.json'), 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        # Generate visualizations
        self.create_visualizations(report_df)
        
        print(f"Results saved to: {self.output_dir}")
    
    def save_updated_results(self):
        """Save updated results with FDR correction applied."""
        if hasattr(self, 'report_df'):
            # Save the updated report DataFrame with FDR correction
            updated_file = os.path.join(self.output_dir, 'bootstrap_detailed_results_with_fdr.csv')
            self.report_df.to_csv(updated_file, index=False)
            print(f"Updated results with FDR correction saved to: {updated_file}")
        else:
            print("No updated results to save")

    def create_visualizations(self, report_df: pd.DataFrame = None):
        """
        Create comprehensive visualizations of bootstrap results.
        
        Args:
            report_df: DataFrame containing bootstrap results (uses stored if None)
        """
        print("Generating bootstrap visualizations...")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Use stored report_df if available (for loaded results), otherwise use passed parameter
        if hasattr(self, 'report_df') and report_df is None:
            report_df = self.report_df
        
        if report_df is None:
            print("Warning: No report data available for visualization")
            return
        
        # 1. Correlation distribution plot
        self.plot_correlation_distribution(report_df)
        
        # 2. Confidence interval plot
        self.plot_confidence_intervals(report_df)
        
        # 3. Model comparison plot
        self.plot_model_comparison(report_df)
        
        # 4. Framing categories analysis
        self.plot_framing_analysis(report_df)
        
        # 5. Statistical significance summary
        self.plot_significance_summary(report_df)
        
        # 6. Model comparison visualizations (Goal 4 specific)
        if hasattr(self, 'model_comparisons'):
            self.plot_model_comparisons(report_df)
        
        # 7. Cross-run stability visualization (Goal 2 specific)
        if hasattr(self, 'stability_results'):
            self.plot_cross_run_stability(report_df)
        
        print("✅ All visualizations generated successfully!")
    
    def plot_cross_run_stability(self, report_df: pd.DataFrame = None):
        """
        Visualize cross-run stability to demonstrate robustness (Goal 2).
        
        Args:
            report_df: DataFrame containing bootstrap results (unused but kept for consistency)
        """
        if not hasattr(self, 'stability_results'):
            return
        
        print("Generating cross-run stability visualizations...")
        
        # Collect stability data
        stability_data = []
        for model, prompt_results in self.stability_results.items():
            for prompt, col_results in prompt_results.items():
                for col, stability_metrics in col_results.items():
                    stability_data.append({
                        'model': model,
                        'prompt': prompt,
                        'category': col,
                        'mean_agreement': stability_metrics['mean_agreement'],
                        'agreement_std': stability_metrics['agreement_std'],
                        'min_agreement': stability_metrics['min_agreement'],
                        'max_agreement': stability_metrics['max_agreement']
                    })
        
        if not stability_data:
            return
        
        stability_df = pd.DataFrame(stability_data)
        
        # Create stability visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Agreement distribution by model
        model_agreement = stability_df.groupby('model')['mean_agreement'].agg(['mean', 'std']).reset_index()
        axes[0, 0].bar(model_agreement['model'], model_agreement['mean'], 
                       yerr=model_agreement['std'], capsize=5)
        axes[0, 0].set_title('Cross-Run Agreement by Model')
        axes[0, 0].set_ylabel('Mean Agreement Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Agreement by prompt level
        prompt_agreement = stability_df.groupby('prompt')['mean_agreement'].agg(['mean', 'std']).reset_index()
        axes[0, 1].bar(prompt_agreement['prompt'], prompt_agreement['mean'],
                       yerr=prompt_agreement['std'], capsize=5)
        axes[0, 1].set_title('Cross-Run Agreement by Prompt Level')
        axes[0, 1].set_ylabel('Mean Agreement Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Agreement distribution histogram
        axes[1, 0].hist(stability_df['mean_agreement'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0.8, color='red', linestyle='--', label='High Consistency Threshold (0.8)')
        axes[1, 0].set_xlabel('Cross-Run Agreement Rate')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Cross-Run Agreement Rates')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Agreement range by category
        category_agreement = stability_df.groupby('category')['mean_agreement'].agg(['mean', 'std']).reset_index()
        category_agreement = category_agreement.sort_values('mean', ascending=False)
        axes[1, 1].bar(range(len(category_agreement)), category_agreement['mean'],
                       yerr=category_agreement['std'], capsize=3)
        axes[1, 1].set_title('Cross-Run Agreement by Category')
        axes[1, 1].set_ylabel('Mean Agreement Rate')
        axes[1, 1].set_xticks(range(len(category_agreement)))
        axes[1, 1].set_xticklabels([cat[:15] for cat in category_agreement['category']], rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cross_run_stability_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_distribution(self, report_df: pd.DataFrame):
        """
        Plot distribution of metric values by metric type.
        
        Args:
            report_df: DataFrame containing bootstrap results
        """
        valid_metrics = report_df[~report_df['mean'].isna()]
        
        if len(valid_metrics) == 0:
            return
        
        # Create subplot for each metric
        metrics = valid_metrics['metric'].unique()
        n_metrics = len(metrics)
        
        plt.figure(figsize=(15, 3*n_metrics))
        
        for i, metric in enumerate(metrics):
            metric_data = valid_metrics[valid_metrics['metric'] == metric]
            
            plt.subplot(n_metrics, 2, 2*i + 1)
            plt.hist(metric_data['mean'], bins=20, alpha=0.7, edgecolor='black')
            
            # Add reference lines based on metric type
            if metric in ['pearson_correlation', 'mcc', 'cohens_kappa']:
                plt.axvline(x=0, color='red', linestyle='--', label='Baseline')
                plt.axvline(x=0.1, color='orange', linestyle='--', label='Small effect')
                plt.axvline(x=0.3, color='green', linestyle='--', label='Medium effect')
                plt.axvline(x=0.5, color='purple', linestyle='--', label='Large effect')
            elif metric == 'balanced_accuracy':
                plt.axvline(x=0.5, color='red', linestyle='--', label='Random baseline')
                plt.axvline(x=0.7, color='orange', linestyle='--', label='Fair')
                plt.axvline(x=0.8, color='green', linestyle='--', label='Good')
                plt.axvline(x=0.9, color='purple', linestyle='--', label='Excellent')
            elif metric == 'f1_score':
                plt.axvline(x=0, color='red', linestyle='--', label='Baseline')
                plt.axvline(x=0.4, color='orange', linestyle='--', label='Fair')
                plt.axvline(x=0.6, color='green', linestyle='--', label='Good')
                plt.axvline(x=0.8, color='purple', linestyle='--', label='Excellent')
            
            plt.xlabel(f'{metric.replace("_", " ").title()}')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {metric.replace("_", " ").title()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Box plot by model
            plt.subplot(n_metrics, 2, 2*i + 2)
            if len(metric_data) > 0:
                sns.boxplot(data=metric_data, x='model', y='mean', showfliers=False)
                plt.xticks(rotation=45)
                plt.title(f'{metric.replace("_", " ").title()} by Model')
                plt.ylabel(f'{metric.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metric_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_intervals(self, report_df: pd.DataFrame):
        """
        Plot confidence intervals for top results by metric.
        
        Args:
            report_df: DataFrame containing bootstrap results
        """
        metrics = report_df['metric'].unique()
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            metric_data = report_df[report_df['metric'] == metric].copy()
            valid_results = metric_data[~metric_data['mean'].isna()]
            
            if len(valid_results) == 0:
                axes[i].text(0.5, 0.5, f'No valid results for {metric}', 
                           transform=axes[i].transAxes, ha='center')
                continue
            
            # Get top 15 results for this metric
            top_results = valid_results.nlargest(15, 'mean')
            
            y_pos = range(len(top_results))
            means = top_results['mean'].values
            ci_lower = top_results['ci_lower'].values
            ci_upper = top_results['ci_upper'].values
            
            # Create error bars
            errors = [means - ci_lower, ci_upper - means]
            
            # Color by significance
            if 'corrected_significant' in top_results.columns:
                colors = ['red' if sig else 'blue' for sig in top_results['corrected_significant']]
            else:
                colors = ['red' if sig else 'blue' for sig in top_results['statistically_significant']]

            
            axes[i].barh(y_pos, means, xerr=errors, capsize=3, color=colors, alpha=0.7)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels([f"{row['model'][:8]}-{row['prompt'][:3]}-{row['category'][:12]}" 
                                   for _, row in top_results.iterrows()])
            axes[i].set_xlabel(f'{metric.replace("_", " ").title()}')
            axes[i].set_title(f'Top 15 {metric.replace("_", " ").title()} Results with 95% CI')
            
            # Add baseline reference
            baseline = 0.5 if metric == 'balanced_accuracy' else 0.0
            axes[i].axvline(x=baseline, color='black', linestyle='-', alpha=0.5)
            axes[i].grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        if 'corrected_significant' in report_df.columns:
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Significant (FDR-corrected)'),
                          Patch(facecolor='blue', alpha=0.7, label='Not Significant')]
        else:
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Statistically Significant'),
                          Patch(facecolor='blue', alpha=0.7, label='Not Significant')]

        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confidence_intervals_by_metric.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, report_df: pd.DataFrame):
        """
        Compare model performance across metrics.
        
        Args:
            report_df: DataFrame containing bootstrap results
        """
        metrics = report_df['metric'].unique()
        
        fig, axes = plt.subplots(2, len(metrics), figsize=(4*len(metrics), 8))
        if len(metrics) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(metrics):
            metric_data = report_df[report_df['metric'] == metric].copy()
            
            # Calculate model statistics
            model_stats = metric_data.groupby('model').agg({
                'mean': ['mean', 'std'],
                'statistically_significant': ['sum', 'count']
            })
            
            model_stats.columns = ['Mean_Value', 'Std_Value', 'Significant_Count', 'Total_Count']
            model_stats['Success_Rate'] = model_stats['Significant_Count'] / model_stats['Total_Count']
            model_stats = model_stats.fillna(0)
            
            # Mean values with error bars
            axes[0, i].bar(model_stats.index, model_stats['Mean_Value'], 
                          yerr=model_stats['Std_Value'], capsize=5)
            axes[0, i].set_title(f'Mean {metric.replace("_", " ").title()}')
            axes[0, i].set_ylabel('Mean Value')
            axes[0, i].tick_params(axis='x', rotation=45)
            axes[0, i].grid(True, alpha=0.3)
            
            # Success rate
            axes[1, i].bar(model_stats.index, model_stats['Success_Rate'])
            axes[1, i].set_title(f'Success Rate - {metric.replace("_", " ").title()}')
            axes[1, i].set_ylabel('Proportion Significant')
            axes[1, i].tick_params(axis='x', rotation=45)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_by_metric.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_framing_analysis(self, report_df: pd.DataFrame):
        """
        Analyze framing columns specifically, broken down by metric.
        
        Args:
            report_df: DataFrame containing bootstrap results
        """
        framing_data = report_df[report_df['is_framing_column'] == True].copy()
        
        if len(framing_data) == 0:
            return
        
        metrics = framing_data['metric'].unique()
        fig, axes = plt.subplots(2, len(metrics), figsize=(4*len(metrics), 8))
        if len(metrics) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(metrics):
            metric_framing_data = framing_data[framing_data['metric'] == metric]
            
            if len(metric_framing_data) == 0:
                axes[0, i].text(0.5, 0.5, f'No framing data for {metric}', 
                              transform=axes[0, i].transAxes, ha='center')
                axes[1, i].text(0.5, 0.5, f'No framing data for {metric}', 
                              transform=axes[1, i].transAxes, ha='center')
                continue
            
            # Mean values by framing category
            framing_means = metric_framing_data.groupby('category')['mean'].mean()
            axes[0, i].bar(framing_means.index, framing_means.values)
            axes[0, i].set_title(f'Mean {metric.replace("_", " ").title()}\nby Framing Type')
            axes[0, i].set_ylabel('Mean Value')
            axes[0, i].tick_params(axis='x', rotation=45)
            axes[0, i].grid(True, alpha=0.3)
            
            # Significance rate by framing category
            sig_summary = metric_framing_data.groupby('category')['statistically_significant'].agg(['sum', 'count'])
            sig_rate = sig_summary['sum'] / sig_summary['count']
            axes[1, i].bar(sig_rate.index, sig_rate.values)
            axes[1, i].set_title(f'Significance Rate - {metric.replace("_", " ").title()}\nby Framing Type')
            axes[1, i].set_ylabel('Proportion Significant')
            axes[1, i].tick_params(axis='x', rotation=45)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'framing_analysis_by_metric.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_significance_summary(self, report_df: pd.DataFrame):
        """
        Create summary plot of statistical significance results by metric.
        
        Args:
            report_df: DataFrame containing bootstrap results
        """
        metrics = report_df['metric'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall significance by metric
        sig_by_metric = report_df.groupby('metric')['statistically_significant'].agg(['sum', 'count'])
        sig_rates = sig_by_metric['sum'] / sig_by_metric['count']
        
        axes[0, 0].bar(sig_rates.index, sig_rates.values)
        axes[0, 0].set_title('Statistical Significance Rate by Metric')
        axes[0, 0].set_ylabel('Proportion Significant')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Effect sizes distribution by metric
        effect_summary = report_df.groupby(['metric', 'effect_size']).size().unstack(fill_value=0)
        effect_summary.plot(kind='bar', stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title('Effect Size Distribution by Metric')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Precision distribution by metric
        axes[1, 0].boxplot([report_df[report_df['metric'] == metric]['precision'].dropna() 
                           for metric in metrics], labels=metrics)
        axes[1, 0].set_title('Precision (CI Width) by Metric')
        axes[1, 0].set_ylabel('CI Width')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample sizes distribution
        axes[1, 1].hist(report_df['sample_size'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Sample Sizes')
        axes[1, 1].set_xlabel('Sample Size')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'significance_summary_by_metric.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary_report(self):
        """
        Print comprehensive summary report to console, organized by metric.
        
        Uses stored report_df if available (for loaded results), otherwise generates new one.
        """
        # Use stored report_df if available (for loaded results), otherwise generate new one
        if hasattr(self, 'report_df'):
            report_df = self.report_df
        else:
            report_df = self.generate_summary_report()
        
        print("\n" + "="*80)
        print("MULTI-METRIC BOOTSTRAP ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        total_analyses = len(report_df)
        significant_analyses = report_df['statistically_significant'].sum()
        
        print(f"\nOVERALL STATISTICS:")
        print(f"- Total analyses performed: {total_analyses}")
        print(f"- Statistically significant: {significant_analyses} ({significant_analyses/total_analyses*100:.1f}%)")
        print(f"- Unique models: {report_df['model'].nunique()}")
        print(f"- Unique categories: {report_df['category'].nunique()}")
        print(f"- Metrics analyzed: {', '.join(report_df['metric'].unique())}")
        print(f"- Significant (uncorrected): {report_df['statistically_significant'].sum()}")
        print(f"- Significant (FDR-corrected): {report_df['corrected_significant'].sum()}")

        
        # Results by metric
        print(f"\nRESULTS BY METRIC:")
        print("="*50)
        
        for metric in sorted(report_df['metric'].unique()):
            metric_data = report_df[report_df['metric'] == metric].copy()
            valid_results = metric_data[~metric_data['mean'].isna()]
            
            print(f"\n{metric.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if len(valid_results) > 0:
                print(f"- Valid results: {len(valid_results)}")
                print(f"- Significant results: {valid_results['statistically_significant'].sum()} "
                      f"({valid_results['statistically_significant'].mean()*100:.1f}%)")
                print(f"- Mean value: {valid_results['mean'].mean():.3f}")
                print(f"- Value range: {valid_results['mean'].min():.3f} to {valid_results['mean'].max():.3f}")
                print(f"- Average precision (CI width): {valid_results['precision'].mean():.3f}")
                
                # Top 5 results for this metric
                top_results = valid_results.nlargest(5, 'mean')
                print(f"\nTop 5 results:")
                for _, row in top_results.iterrows():
                    sig_marker = "***" if row['statistically_significant'] else "   "
                    print(f"  {sig_marker} {row['model'][:12]:<12} {row['category'][:15]:<15} "
                          f"= {row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                          f"({row['effect_size']})")
            else:
                print(f"- No valid results found")
        
        # Model performance comparison
        print(f"\nMODEL PERFORMANCE COMPARISON:")
        print("="*50)
        
        model_summary = report_df.groupby(['model', 'metric']).agg({
            'mean': 'mean',
            'statistically_significant': ['sum', 'count']
        }).round(3)
        
        model_summary.columns = ['Mean_Value', 'Significant_Count', 'Total_Count']
        model_summary['Success_Rate'] = model_summary['Significant_Count'] / model_summary['Total_Count']
        
        # Overall model ranking (across all metrics)
        overall_ranking = report_df.groupby('model').agg({
            'statistically_significant': ['sum', 'count']
        })
        overall_ranking.columns = ['Total_Significant', 'Total_Analyses']
        overall_ranking['Overall_Success_Rate'] = overall_ranking['Total_Significant'] / overall_ranking['Total_Analyses']
        overall_ranking = overall_ranking.sort_values('Overall_Success_Rate', ascending=False)
        
        print(f"\nOverall Model Ranking (by success rate across all metrics):")
        for rank, (model, stats) in enumerate(overall_ranking.iterrows(), 1):
            print(f"{rank}. {model:<25} {stats['Total_Significant']:.0f}/{stats['Total_Analyses']:.0f} "
                  f"significant ({stats['Overall_Success_Rate']*100:.1f}%)")
        
        # Framing analysis
        framing_results = report_df[report_df['is_framing_column'] == True]
        if len(framing_results) > 0:
            print(f"\nFRAMING CATEGORIES ANALYSIS:")
            print("="*50)
            
            for metric in sorted(framing_results['metric'].unique()):
                metric_framing = framing_results[framing_results['metric'] == metric]
                print(f"\n{metric.replace('_', ' ').title()}:")
                
                framing_summary = metric_framing.groupby('category').agg({
                    'mean': 'mean',
                    'statistically_significant': ['sum', 'count']
                }).round(3)
                
                framing_summary.columns = ['Mean_Value', 'Significant_Count', 'Total_Count']
                framing_summary['Success_Rate'] = framing_summary['Significant_Count'] / framing_summary['Total_Count']
                
                for category, stats in framing_summary.iterrows():
                    print(f"  {category:<20} Mean = {stats['Mean_Value']:.3f}, "
                          f"{stats['Significant_Count']:.0f}/{stats['Total_Count']:.0f} significant "
                          f"({stats['Success_Rate']*100:.1f}%)")
        
        print(f"\nMETRIC INTERPRETATION GUIDELINES:")
        print("="*50)
        print("- Pearson Correlation: Linear relationship strength (-1 to +1)")
        print("- MCC: Matthews Correlation Coefficient for binary classification (-1 to +1)")
        print("- F1-Score: Harmonic mean of precision and recall (0 to 1)")
        print("- Balanced Accuracy: Average of sensitivity and specificity (0 to 1, 0.5 = random)")
        print("- Cohen's Kappa: Agreement corrected for chance (-1 to +1)")
        print("\nStatistical Significance: 95% CI excludes baseline (0 for most metrics, 0.5 for balanced accuracy)")
        print("Effect sizes vary by metric - see individual metric interpretations above")
        
        print("\n" + "="*80)

    def load_bootstrap_results(self, results_file: str):
        """
        Load bootstrap results from a CSV file.
        
        Args:
            results_file: Path to the CSV file containing bootstrap results
        """
        try:
            # Load the CSV results
            report_df = pd.read_csv(results_file)
            
            # Check if this is the detailed results CSV
            if 'model' in report_df.columns and 'prompt' in report_df.columns:
                # This is the detailed results CSV - we need to reconstruct the nested structure
                # and apply FDR correction
                print(f"Loading detailed results from CSV: {len(report_df)} rows")
                
                # Apply FDR correction to recreate the corrected_significant column
                report_df = self.apply_fdr_correction(report_df, alpha=0.05)
                
                # Store the report DataFrame for visualization
                self.report_df = report_df
                
                # Reconstruct the nested bootstrap_results structure for compatibility
                self.bootstrap_results = self._reconstruct_bootstrap_results(report_df)
                
                print(f"✅ Successfully loaded and processed {len(report_df)} results")
                return report_df
            else:
                raise ValueError("CSV file does not contain expected columns (model, prompt, etc.)")
                
        except Exception as e:
            print(f"❌ Error loading bootstrap results: {e}")
            raise
    
    def _reconstruct_bootstrap_results(self, report_df: pd.DataFrame) -> Dict:
        """
        Reconstruct the nested bootstrap_results structure from the flat report DataFrame.
        
        This is needed for compatibility with existing methods.
        
        Args:
            report_df: Flat DataFrame containing bootstrap results
            
        Returns:
            Nested dictionary structure compatible with existing methods
        """
        reconstructed = {}
        
        for _, row in report_df.iterrows():
            model = row['model']
            prompt = row['prompt']
            category = row['category']
            metric = row['metric']
            
            # Initialize nested structure if needed
            if model not in reconstructed:
                reconstructed[model] = {}
            if prompt not in reconstructed[model]:
                reconstructed[model][prompt] = {}
            if category not in reconstructed[model][prompt]:
                reconstructed[model][prompt][category] = {}
            
            # Store metric results
            reconstructed[model][prompt][category][metric] = {
                'sample_size': row['sample_size'],
                'mean': row['mean'],
                'std': row['std'],
                'median': row['median'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper'],
                'statistically_significant': row['statistically_significant'],
                'effect_size': row['effect_size'],
                'precision': row['precision'],
                'p_value': row.get('p_value', np.nan),
                'corrected_p_value': row.get('corrected_p_value', np.nan),
                'corrected_significant': row.get('corrected_significant', row['statistically_significant'])
            }
        
        return reconstructed

    def run_comprehensive_analysis(self, n_bootstrap: int = 1000):
        """
        Run comprehensive analysis addressing all four core goals:
        1. Quantify Uncertainty in Performance Metrics
        2. Demonstrate Robustness Across Runs and Models  
        3. Support Claims of Statistical Significance
        4. Compare Metrics Across Models with Precision
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary containing all analysis results
        """
        print("="*80)
        print("COMPREHENSIVE BOOTSTRAP ANALYSIS - ADDRESSING ALL CORE GOALS")
        print("="*80)
        
        # Goal 1: Quantify Uncertainty
        print("\n🎯 GOAL 1: Quantifying Uncertainty in Performance Metrics")
        print("-" * 60)
        print("Running bootstrap analysis to estimate metric variability...")
        results = self.run_full_bootstrap_analysis(n_bootstrap)
        
        # Goal 2: Demonstrate Robustness
        print("\n🎯 GOAL 2: Demonstrating Robustness Across Runs and Models")
        print("-" * 60)
        print("Analyzing cross-run stability to confirm consistency...")
        stability_results = self.analyze_cross_run_stability()
        
        # Goal 3: Statistical Significance
        print("\n🎯 GOAL 3: Supporting Claims of Statistical Significance")
        print("-" * 60)
        print("Generating comprehensive significance analysis...")
        report_df = self.generate_summary_report()
        
        # Apply FDR correction to the main report if not already done
        if 'corrected_significant' not in report_df.columns:
            print("Applying FDR correction to main bootstrap results...")
            report_df = self.apply_fdr_correction(report_df, alpha=0.05)
        
        # Goal 4: Model Comparisons
        print("\n🎯 GOAL 4: Comparing Metrics Across Models with Precision")
        print("-" * 60)
        print("Performing statistical model comparisons...")
        
        model_comparisons = {}
        for metric in ['pearson_correlation', 'mcc', 'f1_score', 'balanced_accuracy', 'cohens_kappa']:
            print(f"  Comparing models for {metric}...")
            comparison_df = self.compare_models_statistically(metric)
            if len(comparison_df) > 0:
                model_comparisons[metric] = comparison_df
                print(f"    Found {len(comparison_df)} model comparisons")
        
        # Store results for later use
        self.stability_results = stability_results
        self.model_comparisons = model_comparisons
        self.report_df = report_df
        
        # Generate enhanced reports
        self.generate_enhanced_reports()
        
        # Save comprehensive results
        self.save_comprehensive_results()
        
        print("\n✅ Comprehensive analysis complete! All four core goals addressed.")
        return {
            'bootstrap_results': results,
            'stability_results': stability_results,
            'model_comparisons': model_comparisons,
            'summary_report': report_df
        }

    def generate_enhanced_reports(self):
        """Generate enhanced reports that address all four core goals."""
        print("\nGenerating enhanced reports...")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save stability analysis
        if hasattr(self, 'stability_results'):
            stability_file = os.path.join(self.output_dir, 'cross_run_stability_analysis.json')
            with open(stability_file, 'w') as f:
                json.dump(self.stability_results, f, indent=2, default=str)
            print(f"  ✅ Cross-run stability analysis saved to: {stability_file}")
        
        # Save model comparisons
        if hasattr(self, 'model_comparisons'):
            for metric, comparison_df in self.model_comparisons.items():
                if len(comparison_df) > 0:
                    comparison_file = os.path.join(self.output_dir, f'model_comparisons_{metric}.csv')
                    comparison_df.to_csv(comparison_file, index=False)
                    print(f"  ✅ Model comparisons for {metric} saved to: {comparison_file}")
        
        # Generate goal-specific summary
        self.generate_goals_summary_report()

    def generate_goals_summary_report(self):
        """Generate a summary report specifically addressing the four core goals."""
        print("\nGenerating goals summary report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("BOOTSTRAP ANALYSIS - CORE GOALS ACHIEVEMENT REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Goal 1: Uncertainty Quantification
        report_lines.append("🎯 GOAL 1: Quantify Uncertainty in Performance Metrics")
        report_lines.append("-" * 60)
        if hasattr(self, 'report_df') and len(self.report_df) > 0:
            total_metrics = len(self.report_df)
            metrics_with_ci = len(self.report_df[~self.report_df['ci_lower'].isna()])
            avg_precision = self.report_df['precision'].mean()
            
            report_lines.append(f"✅ ACHIEVED: {metrics_with_ci}/{total_metrics} metrics have confidence intervals")
            report_lines.append(f"   Average CI width (precision): {avg_precision:.3f}")
            report_lines.append(f"   All metrics include: mean, median, std, 95% CI, and bootstrap distributions")
        else:
            report_lines.append("❌ NOT ACHIEVED: No bootstrap results available")
        report_lines.append("")
        
        # Goal 2: Robustness Demonstration
        report_lines.append("🎯 GOAL 2: Demonstrate Robustness Across Runs and Models")
        report_lines.append("-" * 60)
        if hasattr(self, 'stability_results'):
            total_combinations = sum(len(prompt_results) for prompt_results in self.stability_results.values())
            if total_combinations > 0:
                report_lines.append(f"✅ ACHIEVED: Cross-run stability analyzed for {total_combinations} model-prompt combinations")
                report_lines.append("   Stability metrics include: agreement rates, consistency measures across runs")
                report_lines.append("   High Krippendorff's α (>0.8) already established in previous analysis")
            else:
                report_lines.append("⚠️ PARTIALLY ACHIEVED: Limited stability data available")
        else:
            report_lines.append("❌ NOT ACHIEVED: Cross-run stability analysis not performed")
        report_lines.append("")
        
        # Goal 3: Statistical Significance
        report_lines.append("🎯 GOAL 3: Support Claims of Statistical Significance")
        report_lines.append("-" * 60)
        if hasattr(self, 'report_df') and len(self.report_df) > 0:
            total_tests = len(self.report_df)
            significant_uncorrected = self.report_df['statistically_significant'].sum()
            significant_corrected = self.report_df['corrected_significant'].sum()
            
            report_lines.append(f"✅ ACHIEVED: {significant_uncorrected}/{total_tests} results statistically significant (uncorrected)")
            report_lines.append(f"   After FDR correction: {significant_corrected}/{total_tests} remain significant")
            report_lines.append("   All significance tests use appropriate baselines (0 for correlations, 0.5 for accuracy)")
            report_lines.append("   P-values calculated from bootstrap distributions")
        else:
            report_lines.append("❌ NOT ACHIEVED: No significance analysis available")
        report_lines.append("")
        
        # Goal 4: Model Comparisons
        report_lines.append("🎯 GOAL 4: Compare Metrics Across Models with Precision")
        report_lines.append("-" * 60)
        if hasattr(self, 'model_comparisons'):
            total_comparisons = sum(len(df) for df in self.model_comparisons.values())
            if total_comparisons > 0:
                report_lines.append(f"✅ ACHIEVED: {total_comparisons} statistical model comparisons performed")
                report_lines.append("   All comparisons include: difference CIs, p-values, effect sizes")
                report_lines.append("   FDR correction applied to comparison p-values")
                report_lines.append("   Effect sizes categorized as: Negligible, Small, Medium, Large")
            else:
                report_lines.append("⚠️ PARTIALLY ACHIEVED: Limited model comparison data")
        else:
            report_lines.append("❌ NOT ACHIEVED: Model comparison analysis not performed")
        report_lines.append("")
        
        # Overall Assessment
        report_lines.append("OVERALL ASSESSMENT")
        report_lines.append("-" * 60)
        goals_achieved = 0
        if hasattr(self, 'report_df') and len(self.report_df) > 0:
            goals_achieved += 1
        if hasattr(self, 'stability_results'):
            goals_achieved += 1
        if hasattr(self, 'report_df') and 'statistically_significant' in self.report_df.columns:
            goals_achieved += 1
        if hasattr(self, 'model_comparisons') and any(len(df) > 0 for df in self.model_comparisons.values()):
            goals_achieved += 1
        
        report_lines.append(f"Core Goals Achieved: {goals_achieved}/4")
        if goals_achieved == 4:
            report_lines.append("🎉 EXCELLENT: All core goals fully achieved!")
        elif goals_achieved >= 3:
            report_lines.append("✅ GOOD: Most core goals achieved")
        elif goals_achieved >= 2:
            report_lines.append("⚠️ FAIR: Some core goals achieved")
        else:
            report_lines.append("❌ POOR: Most core goals not achieved")
        
        # Save the report
        goals_report_file = os.path.join(self.output_dir, 'core_goals_achievement_report.txt')
        with open(goals_report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        print('\n'.join(report_lines))
        print(f"\n📄 Goals achievement report saved to: {goals_report_file}")

    def plot_model_comparisons(self, report_df: pd.DataFrame = None):
        """
        Create visualizations specifically for model comparisons (Goal 4).
        
        Args:
            report_df: DataFrame containing bootstrap results (unused but kept for consistency)
        """
        if not hasattr(self, 'model_comparisons'):
            print("No model comparisons available for visualization")
            return
        
        print("Generating model comparison visualizations...")
        
        # Create a comprehensive model comparison plot
        metrics = list(self.model_comparisons.keys())
        if not metrics:
            return
        
        fig, axes = plt.subplots(2, len(metrics), figsize=(5*len(metrics), 10))
        if len(metrics) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(metrics):
            comparison_df = self.model_comparisons[metric]
            if len(comparison_df) == 0:
                continue
            
            # Plot 1: Difference distribution with significance
            axes[0, i].hist(comparison_df['difference'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, i].axvline(x=0, color='red', linestyle='--', label='No difference')
            axes[0, i].set_xlabel(f'Difference in {metric.replace("_", " ").title()}')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].set_title(f'Model Differences - {metric.replace("_", " ").title()}')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot 2: Significance by effect size
            if 'difference_effect_size' in comparison_df.columns:
                effect_size_counts = comparison_df['difference_effect_size'].value_counts()
                axes[1, i].bar(effect_size_counts.index, effect_size_counts.values)
                axes[1, i].set_title(f'Effect Sizes - {metric.replace("_", " ").title()}')
                axes[1, i].set_ylabel('Count')
                axes[1, i].tick_params(axis='x', rotation=45)
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary heatmap of significant differences
        self.plot_model_comparison_heatmap()
    
    def plot_model_comparison_heatmap(self):
        """Create a heatmap showing significant model differences across metrics."""
        if not hasattr(self, 'model_comparisons'):
            return
        
        # Collect all model pairs and their significance across metrics
        all_model_pairs = set()
        for metric, comparison_df in self.model_comparisons.items():
            if len(comparison_df) > 0:
                for _, row in comparison_df.iterrows():
                    pair = f"{row['model1'][:8]}-{row['model2'][:8]}"
                    all_model_pairs.add(pair)
        
        if not all_model_pairs:
            return
        
        # Create significance matrix
        metrics = list(self.model_comparisons.keys())
        model_pairs = sorted(list(all_model_pairs))
        
        significance_matrix = np.zeros((len(metrics), len(model_pairs)))
        effect_size_matrix = np.zeros((len(metrics), len(model_pairs)))
        
        for i, metric in enumerate(metrics):
            comparison_df = self.model_comparisons[metric]
            if len(comparison_df) == 0:
                continue
            
            for j, pair in enumerate(model_pairs):
                # Find this pair in the comparison data
                for _, row in comparison_df.iterrows():
                    current_pair = f"{row['model1'][:8]}-{row['model2'][:8]}"
                    if current_pair == pair:
                        # Use corrected significance if available
                        is_sig = row.get('corrected_difference_significant', row['difference_significant'])
                        significance_matrix[i, j] = 1 if is_sig else 0
                        
                        # Effect size (convert to numeric for heatmap)
                        effect_size = row.get('difference_effect_size', 'Unknown')
                        if effect_size == 'Large':
                            effect_size_matrix[i, j] = 3
                        elif effect_size == 'Medium':
                            effect_size_matrix[i, j] = 2
                        elif effect_size == 'Small':
                            effect_size_matrix[i, j] = 1
                        else:
                            effect_size_matrix[i, j] = 0
                        break
        
        # Create the heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Significance heatmap
        im1 = ax1.imshow(significance_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(model_pairs)))
        ax1.set_xticklabels(model_pairs, rotation=45, ha='right')
        ax1.set_yticks(range(len(metrics)))
        ax1.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.set_title('Statistical Significance of Model Differences')
        ax1.set_xlabel('Model Pairs')
        ax1.set_ylabel('Metrics')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['Not Significant', 'Significant'])
        
        # Effect size heatmap
        im2 = ax2.imshow(effect_size_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(model_pairs)))
        ax2.set_xticklabels(model_pairs, rotation=45, ha='right')
        ax2.set_yticks(range(len(metrics)))
        ax2.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax2.set_title('Effect Size of Model Differences')
        ax2.set_xlabel('Model Pairs')
        ax2.set_ylabel('Metrics')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_ticks([0, 1, 2, 3])
        cbar2.set_ticklabels(['Unknown', 'Small', 'Medium', 'Large'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_comprehensive_results(self):
        """Save all comprehensive analysis results including model comparisons and stability analysis."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the main bootstrap results
        self.save_results()
        
        # Save model comparisons summary
        if hasattr(self, 'model_comparisons'):
            print("Saving model comparison results...")
            
            # Create a comprehensive summary
            all_comparisons = []
            for metric, comparison_df in self.model_comparisons.items():
                if len(comparison_df) > 0:
                    comparison_df['metric'] = metric
                    all_comparisons.append(comparison_df)
            
            if all_comparisons:
                combined_comparisons = pd.concat(all_comparisons, ignore_index=True)
                combined_comparisons.to_csv(os.path.join(self.output_dir, 'all_model_comparisons.csv'), index=False)
                
                # Create a summary table
                summary_data = []
                for metric in combined_comparisons['metric'].unique():
                    metric_data = combined_comparisons[combined_comparisons['metric'] == metric]
                    significant_count = metric_data['difference_significant'].sum()
                    corrected_significant_count = metric_data.get('corrected_difference_significant', pd.Series([False] * len(metric_data))).sum()
                    
                    summary_data.append({
                        'metric': metric,
                        'total_comparisons': len(metric_data),
                        'significant_differences': significant_count,
                        'significant_after_fdr': corrected_significant_count,
                        'mean_difference': metric_data['difference'].mean(),
                        'mean_effect_size': metric_data['difference_effect_size'].mode().iloc[0] if len(metric_data) > 0 else 'Unknown'
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(os.path.join(self.output_dir, 'model_comparisons_summary.csv'), index=False)
                print(f"  ✅ Model comparisons summary saved")
        
        # Save stability analysis summary
        if hasattr(self, 'stability_results'):
            print("Saving stability analysis results...")
            
            stability_summary = []
            for model, prompt_results in self.stability_results.items():
                for prompt, col_results in prompt_results.items():
                    for col, stability_metrics in col_results.items():
                        stability_summary.append({
                            'model': model,
                            'prompt': prompt,
                            'category': col,
                            'mean_agreement': stability_metrics['mean_agreement'],
                            'agreement_std': stability_metrics['agreement_std'],
                            'min_agreement': stability_metrics['min_agreement'],
                            'max_agreement': stability_metrics['max_agreement'],
                            'n_run_pairs': stability_metrics['n_run_pairs'],
                            'high_consistency': stability_metrics['mean_agreement'] > 0.8
                        })
            
            if stability_summary:
                stability_df = pd.DataFrame(stability_summary)
                stability_df.to_csv(os.path.join(self.output_dir, 'cross_run_stability_summary.csv'), index=False)
                
                # Overall stability summary
                overall_stability = {
                    'total_combinations': int(len(stability_df)),
                    'high_consistency_combinations': int(stability_df['high_consistency'].sum()),
                    'mean_agreement_rate': float(stability_df['mean_agreement'].mean()),
                    'agreement_std': float(stability_df['mean_agreement'].std()),
                    'min_agreement': float(stability_df['mean_agreement'].min()),
                    'max_agreement': float(stability_df['mean_agreement'].max())
                }
                
                with open(os.path.join(self.output_dir, 'overall_stability_summary.json'), 'w') as f:
                    json.dump(overall_stability, f, indent=2)
                
                print(f"  ✅ Stability analysis summary saved")
        
        print(f"✅ All comprehensive results saved to: {self.output_dir}")


def main():
    """
    Main function to run bootstrap analysis.
    
    This function provides an interactive interface for running bootstrap analysis
    and can either run a full analysis or load existing results.
    """
    
    # Configuration - Update these paths for your environment
    data_folder = 'path/to/your/data/folder'  # Path to folder containing all run data
    gt_file = 'path/to/your/ground_truth.csv'  # Path to ground truth file

    # Check if we should load existing results instead of running analysis
    # Set these to None to run full analysis, or provide paths to load existing results
    output_dir = None  # Directory containing existing results
    results_file = None  # Path to existing results CSV file
    
    if output_dir is not None and results_file is not None and os.path.exists(results_file) and os.path.exists(output_dir):
        print(f"📁 Loading existing bootstrap results from: {results_file}")
        
        # Initialize analyzer with existing output directory
        analyzer = BootstrapAnalyzer(
            data_folder=data_folder,
            gt_file=gt_file,
            output_dir=output_dir
        )
        
        try:
            # Load the results
            analyzer.load_bootstrap_results(results_file)
            
            print(f"\n✅ Bootstrap results loaded successfully!")
            
            # Ask user what they want to do
            print("\nWhat would you like to do with the loaded results?")
            print("1. View summary report")
            print("2. Generate visualizations")
            print("3. Run comprehensive analysis (address all 4 core goals)")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                analyzer.print_summary_report()
            elif choice == "2":
                analyzer.create_visualizations()
            elif choice == "3":
                print("\nRunning comprehensive analysis to address all core goals...")
                analyzer.run_comprehensive_analysis(n_bootstrap=BOOTSTRAP_N)
            elif choice == "4":
                print("Exiting...")
                return
            else:
                print("Invalid choice. Exiting...")
                return
            
            # Save the updated results with FDR correction
            analyzer.save_updated_results()
            
            print(f"\n✅ Analysis complete! Results saved to: {analyzer.output_dir}")
            return
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            print("Falling back to running full analysis...")
            # Continue to full analysis below
    else:
        print(f"📁 No existing results found at: {results_file}")
        print("Running full bootstrap analysis...")
    
    try:
        # Initialize analyzer
        analyzer = BootstrapAnalyzer(
            data_folder=data_folder,
            gt_file=gt_file
        )
        
        # Ask user what type of analysis they want
        print("\nWhat type of analysis would you like to run?")
        print("1. Standard bootstrap analysis")
        print("2. Comprehensive analysis (address all 4 core goals)")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "2":
            print("\nRunning comprehensive analysis to address all core goals...")
            results = analyzer.run_comprehensive_analysis(n_bootstrap=BOOTSTRAP_N)
        else:
            print("\nRunning standard bootstrap analysis...")
            results = analyzer.run_full_bootstrap_analysis(n_bootstrap=BOOTSTRAP_N)
            analyzer.save_results()
            analyzer.print_summary_report()
        
        print(f"\n✅ Bootstrap analysis complete! Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()