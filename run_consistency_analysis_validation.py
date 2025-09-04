#!/usr/bin/env python3
"""
Run Consistency Analysis for BLM Protest Image Classification
Implements Phase 1 of the Reliability and Robustness Analysis Roadmap

This script analyzes consistency between multiple runs of the same model
to assess reliability of image classification predictions.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Consistency metrics imports
import krippendorff
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("Warning: pingouin not available. ICC calculations will be skipped.")
    print("Install with: pip install pingouin")

class RunConsistencyAnalyzer:
    """Analyzer for measuring consistency between multiple runs of the same model."""
    
    def __init__(self, data_folder: str, gt_file: str, output_dir: Optional[str] = None):
        """
        Initialize the consistency analyzer.
        
        Args:
            data_folder: Path to folder containing all run data
            gt_file: Path to ground truth file (for column reference)
            output_dir: Directory to save outputs (auto-generated if None)
        """
        self.data_folder = data_folder
        self.gt_file = gt_file
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'consistency_analysis_{timestamp}'
        else:
            self.output_dir = output_dir
        
        # Configuration
        self.prompt_levels = ['v2', 'simple', 'detailed', 'expert']
        self.run_count = 3
        self.model_names = ['gemma3_27b_it_q8_0', 'gpt_4_1', 'internvl3_14b', 'internvl3_38b', 'qwen2_5']
        
        # Data containers
        self.binary_columns = []
        self.consistency_results = {}
        self.framing_columns = ['police_solidarity', 'protester_solidarity', 'peace', 'conflict']
        
    def calculate_krippendorff_alpha(self, run1: pd.Series, run2: pd.Series, run3: pd.Series) -> float:
        """Calculate Krippendorff's Alpha for three runs."""
        try:
            # Align indices and convert to integers
            common_index = run1.index.intersection(run2.index).intersection(run3.index)
            run1_aligned = run1[common_index].fillna(False).astype(int)
            run2_aligned = run2[common_index].fillna(False).astype(int)
            run3_aligned = run3[common_index].fillna(False).astype(int)
            
            # Create reliability data matrix (raters x subjects)
            reliability_data = np.vstack([run1_aligned, run2_aligned, run3_aligned])
            
            # Handle edge cases
            unique_values = np.unique(reliability_data)
            if len(unique_values) < 2:
                return 1.0  # Perfect agreement (all same value)
            
            alpha = krippendorff.alpha(
                reliability_data=reliability_data, 
                level_of_measurement='nominal'
            )
            
            return alpha if not np.isnan(alpha) else 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate Krippendorff's Alpha: {str(e)}")
            return np.nan
    
    def calculate_icc(self, run1: pd.Series, run2: pd.Series, run3: pd.Series) -> Dict[str, float]:
        """Calculate Intraclass Correlation Coefficient."""
        if not PINGOUIN_AVAILABLE:
            return {'icc_value': np.nan, 'icc_ci_lower': np.nan, 'icc_ci_upper': np.nan}
        
        try:
            # Align indices
            common_index = run1.index.intersection(run2.index).intersection(run3.index)
            run1_aligned = run1[common_index].fillna(False).astype(int)
            run2_aligned = run2[common_index].fillna(False).astype(int)
            run3_aligned = run3[common_index].fillna(False).astype(int)
            
            # Prepare data for ICC calculation
            data = pd.DataFrame({
                'subject': np.tile(range(len(run1_aligned)), 3),
                'rater': np.repeat(['run1', 'run2', 'run3'], len(run1_aligned)),
                'rating': np.concatenate([run1_aligned, run2_aligned, run3_aligned])
            })
            
            # Calculate ICC(2,1) - two-way random effects, absolute agreement, single measurement
            icc_result = pg.intraclass_corr(data=data, targets='subject', raters='rater', ratings='rating')
            
            # Get ICC(2,1) results (absolute agreement)
            icc_row = icc_result[icc_result['Type'] == 'ICC2']
            if not icc_row.empty:
                return {
                    'icc_value': float(icc_row['ICC'].iloc[0]),
                    'icc_ci_lower': float(icc_row['CI95%'].iloc[0][0]) if isinstance(icc_row['CI95%'].iloc[0], tuple) else np.nan,
                    'icc_ci_upper': float(icc_row['CI95%'].iloc[0][1]) if isinstance(icc_row['CI95%'].iloc[0], tuple) else np.nan
                }
            else:
                return {'icc_value': np.nan, 'icc_ci_lower': np.nan, 'icc_ci_upper': np.nan}
                
        except Exception as e:
            print(f"Warning: Could not calculate ICC: {str(e)}")
            return {'icc_value': np.nan, 'icc_ci_lower': np.nan, 'icc_ci_upper': np.nan}
    
    def calculate_enhanced_percentage_agreements(self, run1: pd.Series, run2: pd.Series, run3: pd.Series) -> Dict[str, float]:
        """Calculate various percentage agreement metrics."""
        # Align indices
        common_index = run1.index.intersection(run2.index).intersection(run3.index)
        run1_aligned = run1[common_index].fillna(False).astype(bool)
        run2_aligned = run2[common_index].fillna(False).astype(bool)
        run3_aligned = run3[common_index].fillna(False).astype(bool)
        
        agreements = {}
        
        try:
            # Overall percentage agreement (all three agree)
            all_agree = (run1_aligned == run2_aligned) & (run2_aligned == run3_aligned)
            agreements['three_way_agreement'] = np.mean(all_agree)
            
            # Majority agreement (at least 2 out of 3 agree)
            run1_eq_run2 = run1_aligned == run2_aligned
            run1_eq_run3 = run1_aligned == run3_aligned
            run2_eq_run3 = run2_aligned == run3_aligned
            majority_agree = run1_eq_run2 | run1_eq_run3 | run2_eq_run3
            agreements['majority_agreement'] = np.mean(majority_agree)
            
            # Pairwise agreements
            agreements['pairwise_12'] = np.mean(run1_aligned == run2_aligned)
            agreements['pairwise_13'] = np.mean(run1_aligned == run3_aligned) 
            agreements['pairwise_23'] = np.mean(run2_aligned == run3_aligned)
            agreements['mean_pairwise'] = np.mean([
                agreements['pairwise_12'], 
                agreements['pairwise_13'], 
                agreements['pairwise_23']
            ])
            
            # Positive agreement (agreement when at least one is True)
            any_true = run1_aligned | run2_aligned | run3_aligned
            if any_true.sum() > 0:
                positive_agree = all_agree[any_true]
                agreements['positive_agreement'] = np.mean(positive_agree) if len(positive_agree) > 0 else np.nan
            else:
                agreements['positive_agreement'] = np.nan
            
            # Negative agreement (agreement when all are False)
            any_false = (~run1_aligned) | (~run2_aligned) | (~run3_aligned)
            if any_false.sum() > 0:
                all_false = (~run1_aligned) & (~run2_aligned) & (~run3_aligned)
                negative_agree = all_false[any_false]
                agreements['negative_agreement'] = np.mean(negative_agree) if len(negative_agree) > 0 else np.nan
            else:
                agreements['negative_agreement'] = np.nan
            
            # Prediction variance (consistency measure)
            run_stack = np.stack([run1_aligned.astype(int), run2_aligned.astype(int), run3_aligned.astype(int)])
            agreements['prediction_variance'] = np.mean(np.var(run_stack, axis=0))
            agreements['prediction_std'] = np.mean(np.std(run_stack, axis=0))
            
        except Exception as e:
            print(f"Warning: Error calculating percentage agreements: {str(e)}")
            # Fill with NaN values for failed calculations
            for key in ['three_way_agreement', 'majority_agreement', 'pairwise_12', 'pairwise_13', 
                       'pairwise_23', 'mean_pairwise', 'positive_agreement', 'negative_agreement',
                       'prediction_variance', 'prediction_std']:
                if key not in agreements:
                    agreements[key] = np.nan
        
        return agreements
    
    def get_binary_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of binary columns, focusing on key categories for consistency analysis."""
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
    
    def load_run_data(self, model: str, prompt: str, run_id: int) -> Optional[pd.DataFrame]:
        """Load data for a specific run."""
        directory = os.path.join(self.data_folder, f'run{run_id}_{model}_{prompt}')
        
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return None
        
        # List all CSV files in the directory
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        if not files:
            print(f"Warning: No CSV files found in {directory}")
            return None
        
        # Load the first CSV file (assuming one prediction file per run)
        try:
            data = pd.read_csv(os.path.join(directory, files[0]))
            return data
        except Exception as e:
            print(f"Error loading {directory}/{files[0]}: {str(e)}")
            return None
    
    def calculate_pairwise_consistency(self, run1: pd.Series, run2: pd.Series, 
                                     label_name: str, model_prompt: str) -> Dict[str, float]:
        """Calculate consistency metrics between two runs for a single label."""
        # Handle NaN values and ensure alignment
        run1_clean = run1.fillna(False).astype(bool)
        run2_clean = run2.fillna(False).astype(bool)
        
        # Align indices
        common_index = run1_clean.index.intersection(run2_clean.index)
        run1_aligned = run1_clean[common_index]
        run2_aligned = run2_clean[common_index]
        
        metrics = {}
        
        try:
            # Cohen's Kappa (accounts for chance agreement)
            metrics['cohens_kappa'] = cohen_kappa_score(run1_aligned, run2_aligned)
            
            # Simple percentage agreement
            metrics['percentage_agreement'] = np.mean(run1_aligned == run2_aligned)
            
            # Positive agreement (agreement when both are True)
            both_true = (run1_aligned == True) & (run2_aligned == True)
            either_true = (run1_aligned == True) | (run2_aligned == True)
            if either_true.sum() > 0:
                metrics['positive_agreement'] = both_true.sum() / either_true.sum()
            else:
                metrics['positive_agreement'] = np.nan
            
            # Negative agreement (agreement when both are False)
            both_false = (run1_aligned == False) & (run2_aligned == False)
            either_false = (run1_aligned == False) | (run2_aligned == False)
            if either_false.sum() > 0:
                metrics['negative_agreement'] = both_false.sum() / either_false.sum()
            else:
                metrics['negative_agreement'] = np.nan
            
            # Sample size for this comparison
            metrics['sample_size'] = len(common_index)
            
        except Exception as e:
            print(f"Warning: Error calculating consistency for {label_name} in {model_prompt}: {str(e)}")
            metrics = {
                'cohens_kappa': np.nan,
                'percentage_agreement': np.nan,
                'positive_agreement': np.nan,
                'negative_agreement': np.nan,
                'sample_size': 0
            }
        
        return metrics
    
    def calculate_three_way_consistency(self, run1: pd.Series, run2: pd.Series, run3: pd.Series,
                                      label_name: str, model_prompt: str) -> Dict[str, float]:
        """Calculate comprehensive consistency metrics across all three runs."""
        # Handle NaN values and ensure alignment
        common_index = run1.index.intersection(run2.index).intersection(run3.index)
        run1_aligned = run1[common_index].fillna(False).astype(bool)
        run2_aligned = run2[common_index].fillna(False).astype(bool)
        run3_aligned = run3[common_index].fillna(False).astype(bool)
        
        metrics = {}
        
        try:
            # Basic sample information
            metrics['sample_size'] = len(common_index)
            
            # Enhanced percentage agreements
            percentage_metrics = self.calculate_enhanced_percentage_agreements(run1_aligned, run2_aligned, run3_aligned)
            metrics.update(percentage_metrics)
            
            # Cohen's Kappa (pairwise) - keeping for comparison
            metrics['kappa_12'] = cohen_kappa_score(run1_aligned, run2_aligned)
            metrics['kappa_13'] = cohen_kappa_score(run1_aligned, run3_aligned)
            metrics['kappa_23'] = cohen_kappa_score(run2_aligned, run3_aligned)
            
            # Mean Cohen's kappa across all pairs
            kappa_scores = [metrics['kappa_12'], metrics['kappa_13'], metrics['kappa_23']]
            valid_kappas = [k for k in kappa_scores if not np.isnan(k)]
            if valid_kappas:
                metrics['mean_cohens_kappa'] = np.mean(valid_kappas)
            else:
                metrics['mean_cohens_kappa'] = np.nan
            
            # Krippendorff's Alpha (primary metric)
            metrics['krippendorff_alpha'] = self.calculate_krippendorff_alpha(run1_aligned, run2_aligned, run3_aligned)
            
            # Intraclass Correlation Coefficient
            icc_results = self.calculate_icc(run1_aligned, run2_aligned, run3_aligned)
            metrics.update(icc_results)
            
            # Additional consistency measures
            run_stack = np.stack([run1_aligned.astype(int), run2_aligned.astype(int), run3_aligned.astype(int)])
            
            # Consistency ratio (proportion of unanimous decisions)
            unanimous_decisions = np.sum(run_stack, axis=0)
            metrics['unanimous_true'] = np.mean(unanimous_decisions == 3)
            metrics['unanimous_false'] = np.mean(unanimous_decisions == 0)
            metrics['unanimous_total'] = metrics['unanimous_true'] + metrics['unanimous_false']
            
            # Variability measures
            metrics['prediction_variance'] = np.mean(np.var(run_stack, axis=0))
            metrics['prediction_std'] = np.mean(np.std(run_stack, axis=0))
            
            # Consistency score (1 - normalized standard deviation)
            max_std = 0.5  # Maximum possible std for binary outcomes
            metrics['consistency_score'] = 1 - (metrics['prediction_std'] / max_std) if max_std > 0 else 1.0
            
        except Exception as e:
            print(f"Warning: Error calculating comprehensive consistency for {label_name} in {model_prompt}: {str(e)}")
            # Fill with NaN values for failed calculations
            default_metrics = {
                'sample_size': 0, 'three_way_agreement': np.nan, 'majority_agreement': np.nan,
                'mean_pairwise': np.nan, 'mean_cohens_kappa': np.nan, 'krippendorff_alpha': np.nan,
                'icc_value': np.nan, 'prediction_std': np.nan, 'consistency_score': np.nan,
                'unanimous_total': np.nan
            }
            metrics.update(default_metrics)
        
        return metrics
    
    def interpret_consistency(self, metric_value: float, metric_type: str = 'kappa') -> str:
        """Interpret consistency metrics according to standard guidelines."""
        if np.isnan(metric_value):
            return "Unable to calculate"
        
        if metric_type in ['kappa', 'krippendorff_alpha']:
            # Landis & Koch (1977) guidelines for kappa
            # Similar thresholds apply to Krippendorff's Alpha
            if metric_value > 0.8:
                return "Excellent reliability"
            elif metric_value > 0.6:
                return "Good reliability"
            elif metric_value > 0.4:
                return "Moderate reliability"
            elif metric_value > 0.2:
                return "Fair reliability"
            else:
                return "Poor reliability"
        
        elif metric_type == 'icc':
            # ICC interpretation guidelines (Cicchetti, 1994)
            if metric_value > 0.9:
                return "Excellent reliability"
            elif metric_value > 0.75:
                return "Good reliability"
            elif metric_value > 0.5:
                return "Moderate reliability"
            elif metric_value > 0.25:
                return "Fair reliability"
            else:
                return "Poor reliability"
        
        elif metric_type == 'agreement':
            # Simple percentage agreement interpretation
            if metric_value > 0.9:
                return "Excellent agreement"
            elif metric_value > 0.8:
                return "Good agreement"
            elif metric_value > 0.7:
                return "Moderate agreement"
            elif metric_value > 0.6:
                return "Fair agreement"
            else:
                return "Poor agreement"
        
        else:
            return "Unknown metric type"
    
    def analyze_model_prompt_consistency(self, model: str, prompt: str) -> Dict[str, Dict]:
        """Analyze consistency for a specific model-prompt combination."""
        print(f"Analyzing consistency for {model} - {prompt}")
        
        # Load all three runs
        run_data = {}
        for run_id in range(self.run_count):
            data = self.load_run_data(model, prompt, run_id)
            if data is not None:
                # Set filename as index for alignment
                if 'filename' in data.columns:
                    data = data.set_index('filename')
                run_data[f'run_{run_id}'] = data
        
        if len(run_data) < 2:
            print(f"Warning: Not enough runs found for {model}-{prompt} (found {len(run_data)})")
            return {}
        
        # Get binary columns from the first available run
        first_run = list(run_data.values())[0]
        binary_columns = self.get_binary_columns(first_run)
        
        if not binary_columns:
            print(f"Warning: No binary columns found for {model}-{prompt}")
            return {}
        
        # Calculate consistency for each binary column
        results = {}
        
        for col in binary_columns:
            # Check if column exists in all runs
            col_exists_in_all = all(col in run_df.columns for run_df in run_data.values())
            if not col_exists_in_all:
                print(f"Warning: Column {col} not found in all runs for {model}-{prompt}")
                continue
            
            if len(run_data) == 3:
                # Three-way analysis
                run_series = [run_data[f'run_{i}'][col] for i in range(3)]
                three_way_metrics = self.calculate_three_way_consistency(
                    run_series[0], run_series[1], run_series[2], 
                    col, f"{model}-{prompt}"
                )
                results[col] = three_way_metrics
                results[col]['analysis_type'] = 'three_way'
                
            elif len(run_data) == 2:
                # Pairwise analysis
                run_keys = list(run_data.keys())
                pairwise_metrics = self.calculate_pairwise_consistency(
                    run_data[run_keys[0]][col], run_data[run_keys[1]][col],
                    col, f"{model}-{prompt}"
                )
                results[col] = pairwise_metrics
                results[col]['analysis_type'] = 'pairwise'
        
        return results
    
    def run_full_analysis(self):
        """Run consistency analysis for all model-prompt combinations."""
        print("Starting comprehensive consistency analysis...")
        print(f"Models: {self.model_names}")
        print(f"Prompt levels: {self.prompt_levels}")
        print(f"Expected runs per combination: {self.run_count}")
        
        # Load ground truth to get binary columns reference
        try:
            gt_df = pd.read_csv(self.gt_file)
            self.binary_columns = self.get_binary_columns(gt_df)
            print(f"Reference binary columns: {len(self.binary_columns)} found")
        except Exception as e:
            print(f"Warning: Could not load ground truth file: {e}")
        
        # Track fallback usage across all analyses
        total_fallback_used = 0
        total_primary_used = 0
        
        # Analyze each model-prompt combination
        for model in self.model_names:
            self.consistency_results[model] = {}
            
            for prompt in self.prompt_levels:
                model_prompt_results = self.analyze_model_prompt_consistency(model, prompt)
                
                if model_prompt_results:
                    self.consistency_results[model][prompt] = model_prompt_results
                    
                    # Primary: Krippendorff's Alpha, fallback: Cohen's kappa
                    primary_alphas = []
                    fallback_kappas = []

                    for col, metrics in model_prompt_results.items():
                        if 'krippendorff_alpha' in metrics and not np.isnan(metrics['krippendorff_alpha']):
                            primary_alphas.append(metrics['krippendorff_alpha'])
                        elif 'mean_cohens_kappa' in metrics and not np.isnan(metrics['mean_cohens_kappa']):
                            fallback_kappas.append(metrics['mean_cohens_kappa'])

                    if primary_alphas:
                        overall_metric = np.mean(primary_alphas)
                        metric_name = "Krippendorff's α"
                        interpretation = self.interpret_consistency(overall_metric, 'krippendorff_alpha')
                        print(f"  {model}-{prompt}: Overall {metric_name} = {overall_metric:.3f} ({interpretation})")
                        total_primary_used += 1
                    elif fallback_kappas:
                        overall_metric = np.mean(fallback_kappas)
                        metric_name = "Cohen's κ"
                        interpretation = self.interpret_consistency(overall_metric, 'kappa')
                        print(f"  {model}-{prompt}: Overall {metric_name} = {overall_metric:.3f} ({interpretation})")
                        total_fallback_used += 1
                    else:
                        print(f"  {model}-{prompt}: Unable to calculate consistency")
                else:
                    print(f"  {model}-{prompt}: No results (insufficient data)")
        
        print(f"\nConsistency analysis complete!")
        
        # Report on metric usage
        if total_fallback_used > 0:
            print(f"Fallback kappas activated for {total_fallback_used} model-prompt combinations")
        else:
            print("Fallback kappas not activated - all Krippendorff's α calculations were successful")
        
        if total_primary_used > 0:
            print(f"Primary Krippendorff's α used for {total_primary_used} model-prompt combinations")
        
        return self.consistency_results
    
    def generate_summary_statistics(self) -> Dict[str, Dict]:
        """Generate summary statistics across all analyses."""
        summary_stats = {}
        
        for model, prompt_results in self.consistency_results.items():
            model_stats = {
                'total_prompts_analyzed': len(prompt_results),
                'framing_consistency': {},
                'overall_consistency': {}
            }
            
            # Collect all metric values across prompts and columns
            all_krippendorff = []
            all_cohens_kappa = []
            all_icc = []
            all_agreements = []
            framing_krippendorff = []
            framing_cohens_kappa = []
            framing_icc = []
            framing_agreements = []
            
            for prompt, col_results in prompt_results.items():
                prompt_krippendorff = []
                prompt_kappa = []
                prompt_icc = []
                prompt_agreement = []
                
                for col, metrics in col_results.items():
                    # Krippendorff's Alpha (primary metric)
                    if 'krippendorff_alpha' in metrics and not np.isnan(metrics['krippendorff_alpha']):
                        alpha = metrics['krippendorff_alpha']
                        all_krippendorff.append(alpha)
                        prompt_krippendorff.append(alpha)
                        
                        if col in self.framing_columns:
                            framing_krippendorff.append(alpha)
                    
                    # Cohen's Kappa (for comparison)
                    if 'mean_cohens_kappa' in metrics and not np.isnan(metrics['mean_cohens_kappa']):
                        kappa = metrics['mean_cohens_kappa']
                        all_cohens_kappa.append(kappa)
                        prompt_kappa.append(kappa)
                        
                        if col in self.framing_columns:
                            framing_cohens_kappa.append(kappa)
                    
                    # ICC
                    if 'icc_value' in metrics and not np.isnan(metrics['icc_value']):
                        icc = metrics['icc_value']
                        all_icc.append(icc)
                        prompt_icc.append(icc)
                        
                        if col in self.framing_columns:
                            framing_icc.append(icc)
                    
                    # Three-way agreement
                    if 'three_way_agreement' in metrics and not np.isnan(metrics['three_way_agreement']):
                        agreement = metrics['three_way_agreement']
                        all_agreements.append(agreement)
                        prompt_agreement.append(agreement)
                        
                        if col in self.framing_columns:
                            framing_agreements.append(agreement)
                
                # Prompt-level statistics
                if prompt_krippendorff:
                    model_stats[f'{prompt}_krippendorff_alpha'] = np.mean(prompt_krippendorff)
                    model_stats[f'{prompt}_krippendorff_interpretation'] = self.interpret_consistency(
                        np.mean(prompt_krippendorff), 'krippendorff_alpha'
                    )
                
                if prompt_kappa:
                    model_stats[f'{prompt}_cohens_kappa'] = np.mean(prompt_kappa)
                    
                if prompt_icc:
                    model_stats[f'{prompt}_icc'] = np.mean(prompt_icc)
                    
                if prompt_agreement:
                    model_stats[f'{prompt}_three_way_agreement'] = np.mean(prompt_agreement)
            
            # Overall model statistics
            if all_krippendorff:
                model_stats['overall_consistency']['krippendorff_alpha'] = np.mean(all_krippendorff)
                model_stats['overall_consistency']['krippendorff_std'] = np.std(all_krippendorff)
                model_stats['overall_consistency']['krippendorff_min'] = np.min(all_krippendorff)
                model_stats['overall_consistency']['krippendorff_max'] = np.max(all_krippendorff)
                model_stats['overall_consistency']['krippendorff_interpretation'] = self.interpret_consistency(
                    np.mean(all_krippendorff), 'krippendorff_alpha'
                )
            
            if all_cohens_kappa:
                model_stats['overall_consistency']['cohens_kappa'] = np.mean(all_cohens_kappa)
                model_stats['overall_consistency']['cohens_kappa_std'] = np.std(all_cohens_kappa)
                
            if all_icc:
                model_stats['overall_consistency']['icc'] = np.mean(all_icc)
                model_stats['overall_consistency']['icc_std'] = np.std(all_icc)
                
            if all_agreements:
                model_stats['overall_consistency']['three_way_agreement'] = np.mean(all_agreements)
                model_stats['overall_consistency']['agreement_std'] = np.std(all_agreements)
            
            # Framing-specific statistics
            if framing_krippendorff:
                model_stats['framing_consistency']['krippendorff_alpha'] = np.mean(framing_krippendorff)
                model_stats['framing_consistency']['krippendorff_interpretation'] = self.interpret_consistency(
                    np.mean(framing_krippendorff), 'krippendorff_alpha'
                )
                
            if framing_cohens_kappa:
                model_stats['framing_consistency']['cohens_kappa'] = np.mean(framing_cohens_kappa)
                
            if framing_icc:
                model_stats['framing_consistency']['icc'] = np.mean(framing_icc)
                
            if framing_agreements:
                model_stats['framing_consistency']['three_way_agreement'] = np.mean(framing_agreements)
            
            summary_stats[model] = model_stats
        
        return summary_stats
    
    def save_results(self):
        """Save all results to files."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save detailed results as JSON
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
        
        detailed_results = convert_numpy(self.consistency_results)
        with open(os.path.join(self.output_dir, 'detailed_consistency_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Generate and save summary statistics
        summary_stats = self.generate_summary_statistics()
        summary_stats = convert_numpy(summary_stats)
        with open(os.path.join(self.output_dir, 'summary_statistics.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Save CSV reports
        self.save_csv_reports()
        
        # Generate visualizations
        self.create_visualizations()
        
        print(f"All results saved to: {self.output_dir}")
    
    def save_csv_reports(self):
        """Save results in CSV format for easy analysis."""
        
        # Detailed results CSV
        detailed_data = []
        for model, prompt_results in self.consistency_results.items():
            for prompt, col_results in prompt_results.items():
                for col, metrics in col_results.items():
                    row = {
                        'model': model,
                        'prompt': prompt,
                        'column': col,
                        'is_framing_column': col in self.framing_columns
                    }
                    row.update(metrics)
                    detailed_data.append(row)
        
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv(os.path.join(self.output_dir, 'detailed_consistency_results.csv'), index=False)
        
        # Summary statistics CSV
        summary_stats = self.generate_summary_statistics()
        summary_data = []
        for model, stats in summary_stats.items():
            row = {'model': model}
            # Flatten nested dictionaries
            for key, value in stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f'{key}_{subkey}'] = subvalue
                else:
                    row[key] = value
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.output_dir, 'summary_statistics.csv'), index=False)
    
    def create_metric_comparison_plot(self):
        """Create plot comparing Krippendorff's Alpha vs Cohen's Kappa where both are available."""
        comparison_data = []
        
        for model, prompt_results in self.consistency_results.items():
            for prompt, col_results in prompt_results.items():
                for col, metrics in col_results.items():
                    if ('krippendorff_alpha' in metrics and not np.isnan(metrics['krippendorff_alpha']) and
                        'mean_cohens_kappa' in metrics and not np.isnan(metrics['mean_cohens_kappa'])):
                        
                        comparison_data.append({
                            'model': model,
                            'prompt': prompt,
                            'column': col,
                            'krippendorff_alpha': metrics['krippendorff_alpha'],
                            'cohens_kappa': metrics['mean_cohens_kappa'],
                            'is_framing': col in self.framing_columns
                        })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            plt.figure(figsize=(10, 8))
            
            # Scatter plot comparing the two metrics
            colors = ['red' if framing else 'blue' for framing in df['is_framing']]
            plt.scatter(df['cohens_kappa'], df['krippendorff_alpha'], 
                    c=colors, alpha=0.6, s=50)
            
            # Add diagonal line (perfect agreement)
            min_val = min(df['cohens_kappa'].min(), df['krippendorff_alpha'].min())
            max_val = max(df['cohens_kappa'].max(), df['krippendorff_alpha'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
            
            plt.xlabel('Cohen\'s κ')
            plt.ylabel('Krippendorff\'s α')
            plt.title('Comparison of Reliability Metrics')
            plt.legend(['Perfect Agreement', 'Framing Columns', 'Other Columns'])
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'metric_comparison.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()

    def create_visualizations(self):
        """Create visualizations of consistency results."""
        print("Generating visualizations...")
        
        # Model consistency heatmap
        self.create_model_consistency_heatmap()
        
        # Framing columns consistency comparison
        self.create_framing_consistency_plot()
        
        # Distribution of kappa values
        self.create_kappa_distribution_plot()

        # Comparison of Krippendorff's Alpha vs Cohen's Kappa
        self.create_metric_comparison_plot()
    
    def create_model_consistency_heatmap(self):
        """Create heatmap showing consistency across models and prompts."""
        # Prepare data for heatmap
        heatmap_data = []
        
        for model, prompt_results in self.consistency_results.items():
            for prompt, col_results in prompt_results.items():
                # Primary: Krippendorff's Alpha, fallback: Cohen's kappa
                alphas = [metrics.get('krippendorff_alpha', np.nan) 
                        for metrics in col_results.values() 
                        if 'krippendorff_alpha' in metrics and not np.isnan(metrics.get('krippendorff_alpha', np.nan))]

                kappas = [metrics.get('mean_cohens_kappa', np.nan) 
                        for metrics in col_results.values() 
                        if 'mean_cohens_kappa' in metrics and not np.isnan(metrics.get('mean_cohens_kappa', np.nan))]

                # Use alphas if available, otherwise kappas
                valid_metrics = alphas if alphas else kappas
                avg_metric = np.mean(valid_metrics) if valid_metrics else np.nan
                
                heatmap_data.append({
                    'model': model,
                    'prompt': prompt,
                    'mean_kappa': avg_metric
                })
        
        if heatmap_data:
            df = pd.DataFrame(heatmap_data)
            pivot_df = df.pivot(index='model', columns='prompt', values='mean_kappa')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                    center=0.6, vmin=0, vmax=1)
            plt.title('Model Consistency by Model and Prompt\n(Krippendorff\'s α')
            plt.xlabel('Prompt Level')
            plt.ylabel('Model')
            
            # Add colorbar label
            cbar = plt.gca().collections[0].colorbar
            cbar.set_label('Reliability Coefficient', rotation=270, labelpad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_consistency_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_framing_consistency_plot(self):
        """Create plot showing consistency for framing columns specifically."""
        framing_data = []
        
        for model, prompt_results in self.consistency_results.items():
            for prompt, col_results in prompt_results.items():
                for col in self.framing_columns:
                    if col in col_results:
                        metrics = col_results[col]
                        # Primary: Krippendorff's Alpha, fallback: Cohen's kappa
                        if 'krippendorff_alpha' in metrics and not np.isnan(metrics['krippendorff_alpha']):
                            framing_data.append({
                                'model': model,
                                'prompt': prompt,
                                'framing_type': col,
                                'alpha': metrics['krippendorff_alpha']
                            })
                        elif 'mean_cohens_kappa' in metrics and not np.isnan(metrics['mean_cohens_kappa']):
                            framing_data.append({
                                'model': model,
                                'prompt': prompt,
                                'framing_type': col,
                                'alpha': metrics['mean_cohens_kappa']
                            })
        
        if framing_data:
            df = pd.DataFrame(framing_data)
            
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=df, x='model', y='alpha', hue='framing_type', showfliers=False)
            plt.title('Consistency for Framing Columns by Model (Krippendorff\'s α)')
            plt.xlabel('Model')
            plt.ylabel('Reliability Coefficient')
            plt.xticks(rotation=45)
            plt.legend(title='Framing Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add note about metrics
            plt.figtext(0.02, 0.02, 'Primary: Krippendorff\'s α, Fallback: Cohen\'s κ', fontsize=8, style='italic')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'framing_consistency_boxplot.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_kappa_distribution_plot(self):
        """Create distribution plot of all reliability coefficient values."""
        all_metrics = []
        
        for model, prompt_results in self.consistency_results.items():
            for prompt, col_results in prompt_results.items():
                for col, metrics in col_results.items():
                    metric_value = None
                    metric_type = None
                    
                    if 'krippendorff_alpha' in metrics and not np.isnan(metrics['krippendorff_alpha']):
                        metric_value = metrics['krippendorff_alpha']
                        metric_type = 'Krippendorff\'s α'
                    elif 'mean_cohens_kappa' in metrics and not np.isnan(metrics['mean_cohens_kappa']):
                        metric_value = metrics['mean_cohens_kappa']
                        metric_type = 'Cohen\'s κ'

                    if metric_value is not None:
                        all_metrics.append({
                            'model': model,
                            'reliability_coefficient': metric_value,
                            'metric_type': metric_type,
                            'column_type': 'framing' if col in self.framing_columns else 'other'
                        })
        
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            
            plt.figure(figsize=(12, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(df['reliability_coefficient'], bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=0.6, color='orange', linestyle='--', label='Good threshold (0.6)')
            plt.axvline(x=0.8, color='green', linestyle='--', label='Excellent threshold (0.8)')
            plt.xlabel('Reliability Coefficient')
            plt.ylabel('Frequency')
            plt.title('Distribution of Reliability Coefficients')
            plt.legend()
            
            # Box plot by model
            plt.subplot(1, 2, 2)
            sns.boxplot(data=df, x='model', y='reliability_coefficient', showfliers=False)
            plt.xticks(rotation=45)
            plt.title('Reliability Distribution by Model')
            plt.ylabel('Reliability Coefficient')
            
            # Add note about metrics
            # plt.figtext(0.02, 0.02, 'Primary: Krippendorff\'s α, fontsize=8, style='italic')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'reliability_distribution.png'),
                    dpi=300, bbox_inches='tight')
            plt.close()
            
    def print_summary_report(self):
        """Print comprehensive summary report."""
        summary_stats = self.generate_summary_statistics()
        
        print("\n" + "="*80)
        print("RUN CONSISTENCY ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\nANALYSIS SCOPE:")
        print(f"- Models analyzed: {len(self.consistency_results)}")
        print(f"- Prompt levels: {self.prompt_levels}")
        print(f"- Expected runs per combination: {self.run_count}")
        
        # Model rankings by overall consistency
        print(f"\nMODEL CONSISTENCY RANKINGS:")
        print("-" * 50)
        
        model_rankings = []
        for model, stats in summary_stats.items():
            primary_metric = stats.get('overall_consistency', {}).get('krippendorff_alpha', np.nan)
            if np.isnan(primary_metric):
                primary_metric = stats.get('overall_consistency', {}).get('cohens_kappa', np.nan)
            if not np.isnan(primary_metric): 
                model_rankings.append((model, primary_metric))
        
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, metric_value) in enumerate(model_rankings, 1):
            interpretation = self.interpret_consistency(metric_value, 'krippendorff_alpha')
            print(f"{rank:2d}. {model:<25} α/κ = {metric_value:.3f} ({interpretation})")

        
        # Detailed model statistics
        print(f"\nDETAILED MODEL STATISTICS:")
        print("-" * 50)
        
        for model, stats in summary_stats.items():
            print(f"\n{model}:")
            
            # Overall consistency
            overall = stats.get('overall_consistency', {})
            if overall:
                print(f"  Overall Consistency:")
                
                # Primary metrics
                if 'krippendorff_alpha' in overall:
                    alpha_val = overall['krippendorff_alpha']
                    alpha_std = overall.get('krippendorff_std', 0)
                    alpha_min = overall.get('krippendorff_min', alpha_val)
                    alpha_max = overall.get('krippendorff_max', alpha_val)
                    alpha_interp = overall.get('krippendorff_interpretation', 'N/A')
                    
                    print(f"    Krippendorff's α: {alpha_val:.3f} ± {alpha_std:.3f}")
                    print(f"    α Range:         {alpha_min:.3f} - {alpha_max:.3f}")
                    print(f"    α Interpretation: {alpha_interp}")
                
                if 'cohens_kappa' in overall:
                    kappa_val = overall['cohens_kappa']
                    kappa_std = overall.get('cohens_kappa_std', 0)
                    print(f"    Cohen's κ:       {kappa_val:.3f} ± {kappa_std:.3f}")
                
                if 'icc' in overall:
                    icc_val = overall['icc']
                    icc_std = overall.get('icc_std', 0)
                    print(f"    ICC:             {icc_val:.3f} ± {icc_std:.3f}")
                
                if 'three_way_agreement' in overall:
                    agree_val = overall['three_way_agreement']
                    agree_std = overall.get('agreement_std', 0)
                    print(f"    3-way Agreement: {agree_val:.3f} ± {agree_std:.3f}")
            
            # Framing consistency
            framing = stats.get('framing_consistency', {})
            if framing:
                print(f"  Framing Categories:")
                
                if 'krippendorff_alpha' in framing:
                    framing_alpha = framing['krippendorff_alpha']
                    framing_interp = framing.get('krippendorff_interpretation', 'N/A')
                    print(f"    Krippendorff's α: {framing_alpha:.3f} ({framing_interp})")
                
                if 'cohens_kappa' in framing:
                    framing_kappa = framing['cohens_kappa']
                    print(f"    Cohen's κ:       {framing_kappa:.3f}")
                
                if 'icc' in framing:
                    framing_icc = framing['icc']
                    print(f"    ICC:             {framing_icc:.3f}")
            
            # Prompt-specific results (showing Krippendorff's Alpha)
            prompt_results = []
            for key, value in stats.items():
                if key.endswith('_krippendorff_alpha'):
                    prompt = key.replace('_krippendorff_alpha', '')
                    interpretation_key = f'{prompt}_krippendorff_interpretation'
                    interpretation = stats.get(interpretation_key, 'N/A')
                    prompt_results.append((prompt, value, interpretation))
            
            if prompt_results:
                print(f"  Prompt-specific (Krippendorff's α):")
                for prompt, alpha, interpretation in prompt_results:
                    print(f"    {prompt:<10}: α = {alpha:.3f} ({interpretation})")
        
        # Recommendations based on results
        print(f"\nRECOMMENDations FOR BOOTSTRAP ANALYSIS:")
        print("-" * 50)
        
        high_consistency_models = []
        moderate_consistency_models = []
        low_consistency_models = []
        
        for model, stats in summary_stats.items():
            # Primary metric for decision making
            primary_metric = stats.get('overall_consistency', {}).get('krippendorff_alpha', np.nan)
            if np.isnan(primary_metric):
                primary_metric = stats.get('overall_consistency', {}).get('cohens_kappa', np.nan)
            
            if not np.isnan(primary_metric):
                print(f"Model: {model}, Primary metric: {primary_metric}")
                if primary_metric > 0.8:
                    high_consistency_models.append(model)
                elif primary_metric >= 0.6:
                    moderate_consistency_models.append(model)
                else:
                    low_consistency_models.append(model)

        
        if high_consistency_models:
            print(f"HIGH CONSISTENCY (κ > 0.8) - Use Bootstrap Option A:")
            for model in high_consistency_models:
                print(f"   - {model}")
        
        if moderate_consistency_models:
            print(f"MODERATE CONSISTENCY (0.6 ≤ κ ≤ 0.8) - Use Bootstrap Option B:")
            for model in moderate_consistency_models:
                print(f"   - {model}")
        
        if low_consistency_models:
            print(f"LOW CONSISTENCY (κ < 0.6) - Use Bootstrap Option C + Model Investigation:")
            for model in low_consistency_models:
                print(f"   - {model}")
        
        print("\n" + "="*80)


def main():
    """Main function to run consistency analysis.
    
    This function initializes the RunConsistencyAnalyzer, runs the full analysis,
    saves results, and prints a comprehensive summary report.
    
    Note: Update the data_folder and gt_file paths before running.
    """
    
    # Configuration - Update these paths for your environment
    data_folder = 'path/to/your/run/data/folder'
    gt_file = 'path/to/your/ground/truth/file.csv'
    
    try:
        # Initialize analyzer
        analyzer = RunConsistencyAnalyzer(
            data_folder=data_folder,
            gt_file=gt_file
        )
        
        # Run full analysis
        results = analyzer.run_full_analysis()
        
        # Save results
        analyzer.save_results()
        
        # Print summary report
        analyzer.print_summary_report()
        
        print(f"\nConsistency analysis complete! Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()