# metaphlan_tools/stats.py

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import skbio.diversity.alpha as alpha_diversity
import skbio.diversity.beta as beta_diversity
from skbio.stats.ordination import pcoa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def calculate_alpha_diversity(abundance_df, metrics=None):
    """
    Calculate alpha diversity metrics for each sample.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metrics : list, optional
        List of alpha diversity metrics to calculate (default: Shannon, Simpson, Observed)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with alpha diversity metrics for each sample
    """
    if metrics is None:
        metrics = ['shannon', 'simpson', 'observed_otus']
    
    # Transpose to have samples as rows
    abundance = abundance_df.T
    
    # Ensure all values are numeric
    abundance = abundance.apply(pd.to_numeric, errors='coerce')
    
    # Replace any NaN values with zeros
    abundance = abundance.fillna(0)
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=abundance.index)
    
    # Calculate each metric
    for metric in metrics:
        if metric == 'shannon':
            # Use try-except to catch and handle errors
            try:
                results['Shannon'] = abundance.apply(
                    lambda x: alpha_diversity.shannon(x) if not x.isna().any() else np.nan, 
                    axis=1
                )
            except Exception as e:
                print(f"Error calculating Shannon diversity: {str(e)}")
                results['Shannon'] = np.nan
                
        elif metric == 'simpson':
            try:
                results['Simpson'] = abundance.apply(
                    lambda x: alpha_diversity.simpson(x) if not x.isna().any() else np.nan, 
                    axis=1
                )
            except Exception as e:
                print(f"Error calculating Simpson diversity: {str(e)}")
                results['Simpson'] = np.nan
                
        elif metric == 'observed_otus':
            results['Observed'] = abundance.apply(lambda x: (x > 0).sum(), axis=1)
    
    return results


def compare_alpha_diversity(alpha_df, metadata_df, variable):
    """
    Compare alpha diversity metrics between groups defined by a metadata variable.
    
    Parameters:
    -----------
    alpha_df : pandas.DataFrame
        Alpha diversity DataFrame with samples as index
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    variable : str
        Metadata variable to group by
        
    Returns:
    --------
    dict
        Dictionary with statistical test results for each metric
    """
    if variable not in metadata_df.columns:
        raise ValueError(f"Variable '{variable}' not found in metadata")
    
    # Join alpha diversity with metadata
    merged = alpha_df.join(metadata_df[[variable]], how='inner')
    
    # Initialize results
    results = {}
    
    # Test each metric
    for metric in alpha_df.columns:
        # Skip if all values are identical
        if merged[metric].nunique() <= 1:
            results[metric] = {
                'test': 'None',
                'statistic': None,
                'p-value': None,
                'note': 'All values are identical, no statistical test performed'
            }
            continue
            
        # For binary variables
        unique_values = merged[variable].nunique()
        
        if unique_values == 2:
            # Use Mann-Whitney U test
            groups = merged.groupby(variable)[metric].apply(list).to_dict()
            group_values = list(groups.values())
            
            # Check if any group has all identical values
            identical_values = any(len(set(group)) == 1 for group in group_values)
            
            if identical_values:
                results[metric] = {
                    'test': 'Mann-Whitney U',
                    'statistic': None,
                    'p-value': None,
                    'note': 'Some groups have identical values, test could not be performed'
                }
            else:
                try:
                    stat, pval = stats.mannwhitneyu(*group_values)
                    test_name = 'Mann-Whitney U'
                    results[metric] = {
                        'test': test_name,
                        'statistic': stat,
                        'p-value': pval
                    }
                except Exception as e:
                    results[metric] = {
                        'test': 'Mann-Whitney U',
                        'statistic': None,
                        'p-value': None,
                        'note': f'Error performing test: {str(e)}'
                    }
                
        elif 2 < unique_values <= 10:
            # Use Kruskal-Wallis test
            group_data = [group[metric].values for name, group in merged.groupby(variable)]
            
            # Check if all values in any group are identical
            identical_values = any(len(set(group)) == 1 for group in group_data)
            
            # Check if all values across all groups are identical
            all_identical = len(set(np.concatenate(group_data))) == 1
            
            if all_identical:
                results[metric] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': None,
                    'p-value': None,
                    'note': 'All values are identical across groups, test could not be performed'
                }
            else:
                try:
                    stat, pval = stats.kruskal(*group_data)
                    test_name = 'Kruskal-Wallis'
                    results[metric] = {
                        'test': test_name,
                        'statistic': stat,
                        'p-value': pval
                    }
                except ValueError as e:
                    results[metric] = {
                        'test': 'Kruskal-Wallis',
                        'statistic': None,
                        'p-value': None,
                        'note': f'Error performing test: {str(e)}'
                    }
                
        else:
            # Use correlation for continuous variables
            try:
                stat, pval = stats.spearmanr(merged[metric], merged[variable])
                test_name = 'Spearman correlation'
                results[metric] = {
                    'test': test_name,
                    'statistic': stat,
                    'p-value': pval
                }
            except Exception as e:
                results[metric] = {
                    'test': 'Spearman correlation',
                    'statistic': None,
                    'p-value': None,
                    'note': f'Error performing test: {str(e)}'
                }
    
    return results


def calculate_beta_diversity(abundance_df, metric='braycurtis'):
    """
    Calculate beta diversity distance matrix.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metric : str, optional
        Beta diversity metric to use (default: 'braycurtis')
        
    Returns:
    --------
    skbio.DistanceMatrix
        Distance matrix of beta diversity between samples
    """
    from skbio import DistanceMatrix
    from scipy.spatial.distance import pdist, squareform
    import numpy as np
    
    # Transpose to have samples as rows, species as columns
    abundance = abundance_df.T
    
    # Ensure all values are numeric
    abundance = abundance.apply(pd.to_numeric, errors='coerce')
    
    # Replace any NaN values with zeros
    abundance = abundance.fillna(0)
    
    # Convert to numpy array with float data type
    abundance_array = abundance.values.astype(float)
    
    try:
        # Calculate distance matrix
        if metric == 'braycurtis':
            distances = squareform(pdist(abundance_array, metric='braycurtis'))
        elif metric == 'jaccard':
            # For jaccard, convert to binary presence/absence first
            binary_array = (abundance_array > 0).astype(float)
            distances = squareform(pdist(binary_array, metric='jaccard'))
        elif metric == 'euclidean':
            distances = squareform(pdist(abundance_array, metric='euclidean'))
        else:
            raise ValueError(f"Unsupported beta diversity metric: {metric}")
        
        # Check for NaN values and replace with zeros (happens when rows have all zeros)
        if np.isnan(distances).any():
            print("Warning: NaN values found in distance matrix, replacing with zeros")
            distances = np.nan_to_num(distances)
        
        # Ensure symmetry by averaging with its transpose
        distances = (distances + distances.T) / 2
        
        # Create distance matrix
        dm = DistanceMatrix(distances, ids=abundance.index)
        return dm
    
    except Exception as e:
        print(f"Error calculating beta diversity: {str(e)}")
        # Create a minimal valid distance matrix as fallback
        n = len(abundance.index)
        # Identity matrix (zeros on diagonal, ones elsewhere) ensures a valid distance matrix
        fallback_distances = np.ones((n, n)) - np.eye(n)
        dm = DistanceMatrix(fallback_distances, ids=abundance.index)
        return dm    

def perform_permanova(distance_matrix, metadata_df, variable):
    """
    Perform PERMANOVA test to determine if community composition differs between groups.
    
    Parameters:
    -----------
    distance_matrix : skbio.DistanceMatrix
        Beta diversity distance matrix
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    variable : str
        Metadata variable to test
        
    Returns:
    --------
    dict
        PERMANOVA test results
    """
    from skbio.stats.distance import permanova
    
    # Filter metadata to include only samples in distance matrix
    common_samples = set(distance_matrix.ids).intersection(set(metadata_df.index))
    metadata_filtered = metadata_df.loc[list(common_samples)]
    
    # Reorder distance matrix to match metadata
    distance_matrix = distance_matrix.filter(metadata_filtered.index)
    
    # Verify groups have enough samples
    group_counts = metadata_filtered[variable].value_counts()
    valid_groups = group_counts[group_counts >= 2].index
    
    if len(valid_groups) < 2:
        print(f"Warning: Not enough groups with sufficient samples for PERMANOVA")
        return {
            'test-statistic': float('nan'),
            'p-value': 1.0,
            'sample size': len(metadata_filtered),
            'note': 'Insufficient group sizes for valid test'
        }
    
    # Filter to only include valid groups
    if len(valid_groups) < len(group_counts):
        metadata_filtered = metadata_filtered[metadata_filtered[variable].isin(valid_groups)]
        distance_matrix = distance_matrix.filter(metadata_filtered.index)
    
    try:
        # Perform PERMANOVA
        result = permanova(distance_matrix, metadata_filtered[variable], permutations=999)
        
        return {
            'test-statistic': result['test statistic'],
            'p-value': result['p-value'],
            'sample size': result['sample size']
        }
    except Exception as e:
        print(f"Error in PERMANOVA: {str(e)}")
        return {
            'test-statistic': float('nan'),
            'p-value': float('nan'),
            'sample size': len(metadata_filtered),
            'note': f'Error performing test: {str(e)}'
        }
    
def plot_ordination(beta_dm, metadata_df, var, method='PCoA'):
    """
    Safely create ordination plot with better handling for negative eigenvalues.
    """
    try:
        from skbio.stats.ordination import pcoa
        import seaborn as sns
        
        # Perform PCoA with correction for negative eigenvalues
        pcoa_results = pcoa(beta_dm, method='eigh')  # Use 'eigh' method which better handles negative eigenvalues
        
        # Get the first two principal coordinates
        pc1 = pcoa_results.samples.iloc[:, 0]
        pc2 = pcoa_results.samples.iloc[:, 1]
        
        # Filter metadata to only include samples in the distance matrix
        common_samples = list(set(beta_dm.ids).intersection(set(metadata_df.index)))
        
        # Create a DataFrame for plotting with only common samples
        plot_df = pd.DataFrame({
            'PC1': pc1,
            'PC2': pc2,
            'Sample': beta_dm.ids
        })
        
        # Join with filtered metadata to get the grouping variable
        plot_df = plot_df.set_index('Sample')
        plot_df[var] = metadata_df.loc[common_samples, var]
        
        # Calculate variance explained
        variance_explained = pcoa_results.proportion_explained
        pc1_var = variance_explained[0] * 100
        pc2_var = variance_explained[1] * 100
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue=var, s=100, ax=ax)
        
        # Add axis labels with variance explained
        ax.set_xlabel(f'PC1 ({pc1_var:.1f}% variance explained)')
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}% variance explained)')
        
        # Add title and legend
        ax.set_title(f'{method} of Beta Diversity ({var})')
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating ordination plot: {str(e)}")
        
        # Create a simple error message plot with more diagnostics
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f"Error creating ordination plot:\n{str(e)}",
               ha='center', va='center', fontsize=12)
        ax.set_title(f'{method} of Beta Diversity ({var})')
        ax.axis('off')
        
        # Print more diagnostic information
        print(f"Distance matrix shape: {beta_dm.shape}")
        print(f"Number of samples in metadata with group variable {var}: {metadata_df[var].count()}")
        print(f"Groups in {var}: {metadata_df[var].unique()}")
        
        return fig
    

def plot_beta_diversity_ordination(distance_matrix, metadata_df, variable, method='PCoA'):
    """
    Perform ordination on beta diversity and create a plot.
    
    Parameters:
    -----------
    distance_matrix : skbio.DistanceMatrix
        Beta diversity distance matrix
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    variable : str
        Metadata variable to color points by
    method : str, optional
        Ordination method ('PCoA' or 'NMDS')
        
    Returns:
    --------
    matplotlib.figure.Figure
        Ordination plot
    """
    # Filter metadata to include only samples in distance matrix
    common_samples = set(distance_matrix.ids).intersection(set(metadata_df.index))
    metadata_filtered = metadata_df.loc[list(common_samples)]
    
    # Reorder distance matrix to match metadata
    distance_matrix = distance_matrix.filter(metadata_filtered.index)
    
    # Perform ordination
    if method == 'PCoA':
        try:
            ordination_result = pcoa(distance_matrix)
            
            # Extract first two axes
            pc1 = ordination_result.samples.iloc[:, 0]  # Use iloc instead of ['PC1']
            pc2 = ordination_result.samples.iloc[:, 1]  # Use iloc instead of ['PC2']
            explained_var = ordination_result.proportion_explained
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'PC1': pc1,
                'PC2': pc2,
                variable: metadata_filtered[variable]
            }, index=metadata_filtered.index)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue=variable, data=plot_df, ax=ax)
            
            # Use iloc to access proportion explained values
            ax.set_xlabel(f'PC1 ({explained_var.iloc[0]:.2%} explained)')
            ax.set_ylabel(f'PC2 ({explained_var.iloc[1]:.2%} explained)')
            ax.set_title('PCoA of Beta Diversity')
            
        except Exception as e:
            print(f"Error in ordination: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"Error creating ordination plot:\n{str(e)}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title('PCoA of Beta Diversity')
            ax.axis('off')
            
    else:
        raise ValueError(f"Unsupported ordination method: {method}")
    
    return fig

def differential_abundance_analysis(abundance_df, metadata_df, variable, method='wilcoxon'):
    """
    Identify species that differ significantly in abundance between groups.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    variable : str
        Metadata variable defining groups
    method : str, optional
        Statistical test to use ('wilcoxon', 'kruskal', 'ancom')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with test results for each species
    """
    # Transpose to have samples as rows
    abundance = abundance_df.T
    
    # Join with metadata
    merged = abundance.join(metadata_df[[variable]], how='inner')
    
    # Initialize results
    results = []
    
    if method in ['wilcoxon', 'kruskal']:
        # For each species
        for species in abundance.columns:
            # Skip if all zeros
            if (merged[species] == 0).all():
                continue
                
            # For binary variables
            unique_values = merged[variable].nunique()
            
            if unique_values == 2 and method == 'wilcoxon':
                # Use Mann-Whitney U test (equivalent to Wilcoxon rank-sum for two independent samples)
                groups = merged.groupby(variable)[species].apply(list).to_dict()
                group_values = list(groups.values())
                stat, pval = stats.mannwhitneyu(*group_values)
                test_name = 'Mann-Whitney U'
                
                # Calculate mean abundance in each group
                group_means = merged.groupby(variable)[species].mean().to_dict()
                mean_diff = max(group_means.values()) - min(group_means.values())
                fold_change = max(group_means.values()) / (min(group_means.values()) + 1e-10)  # Avoid division by zero
                
            elif 2 < unique_values <= 10 and method == 'kruskal':
                # Use Kruskal-Wallis test
                stat, pval = stats.kruskal(*[group[species].values for name, group in merged.groupby(variable)])
                test_name = 'Kruskal-Wallis'
                
                # Calculate range of means
                group_means = merged.groupby(variable)[species].mean().to_dict()
                mean_diff = max(group_means.values()) - min(group_means.values())
                fold_change = max(group_means.values()) / (min(group_means.values()) + 1e-10)
                
            else:
                continue
            
            # Add to results
            results.append({
                'Species': species,
                'Test': test_name,
                'Statistic': stat,
                'P-value': pval,
                'Mean Difference': mean_diff,
                'Fold Change': fold_change
            })
    
    else:
        raise ValueError(f"Unsupported differential abundance method: {method}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Perform multiple testing correction
    if not results_df.empty:
        _, corrected_pvals, _, _ = multipletests(results_df['P-value'], method='fdr_bh')
        results_df['Adjusted P-value'] = corrected_pvals
        
        # Sort by adjusted p-value
        results_df = results_df.sort_values('Adjusted P-value')
    
    return results_df


def plot_abundance_boxplot(abundance_df, metadata_df, species, variable):
    """
    Create a boxplot comparing abundance of a species between groups.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    species : str
        Species to plot
    variable : str
        Metadata variable defining groups
        
    Returns:
    --------
    matplotlib.figure.Figure
        Boxplot figure
    """
    # Check if species exists
    if species not in abundance_df.index:
        raise ValueError(f"Species '{species}' not found in abundance data")
    
    # Extract the species data
    species_abundance = abundance_df.loc[species]
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Abundance': species_abundance,
        variable: metadata_df.loc[species_abundance.index, variable]
    })
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=variable, y='Abundance', data=plot_df, ax=ax)
    ax.set_title(f'Abundance of {species} by {variable}')
    ax.set_ylabel('Relative Abundance (%)')
    
    return fig
