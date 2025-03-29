# metaphlan_tools/viz.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.spatial import distance


def plot_relative_abundance_heatmap(abundance_df, metadata_df=None, group_var=None, 
                                   top_n=20, cluster_samples=True, cluster_taxa=True,
                                   cmap='viridis', figsize=(12, 10)):
    """
    Create a heatmap of relative abundance data.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metadata_df : pandas.DataFrame, optional
        Metadata DataFrame with samples as index
    group_var : str, optional
        Metadata variable to group and color samples by
    top_n : int, optional
        Number of most abundant species to include (default: 20)
    cluster_samples : bool, optional
        Whether to cluster samples (default: True)
    cluster_taxa : bool, optional
        Whether to cluster taxa (default: True)
    cmap : str, optional
        Colormap for the heatmap (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        Heatmap figure
    """
    # Copy the data to avoid modifying the original
    abundance = abundance_df.copy()
    
    # Filter to top N most abundant species
    if top_n is not None and top_n < len(abundance.index):
        # Calculate mean abundance across samples for each species
        mean_abundance = abundance.mean(axis=1)
        # Get the top N species
        top_species = mean_abundance.nlargest(top_n).index
        abundance = abundance.loc[top_species]
    
    # Set up plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine row and column clustering
    row_order = abundance.index
    if cluster_taxa:
        # Cluster species (rows)
        row_linkage = hierarchy.linkage(
            distance.pdist(abundance.values), 
            method='average'
        )
        row_order = abundance.index[hierarchy.leaves_list(row_linkage)]
    
    col_order = abundance.columns
    if cluster_samples:
        # Cluster samples (columns)
        col_linkage = hierarchy.linkage(
            distance.pdist(abundance.values.T), 
            method='average'
        )
        col_order = abundance.columns[hierarchy.leaves_list(col_linkage)]
    
    # Reorder data
    abundance = abundance.loc[row_order, col_order]
    
    # Create color bar for metadata if provided
    if metadata_df is not None and group_var is not None:
        # Get metadata for samples in abundance table
        meta_subset = metadata_df.loc[abundance.columns, [group_var]]
        
        # Create row colors
        lut = dict(zip(meta_subset[group_var].unique(), 
                      sns.color_palette("Set2", meta_subset[group_var].nunique())))
        row_colors = meta_subset[group_var].map(lut)
        
        # Plot heatmap with row colors
        g = sns.clustermap(
            abundance.T,  # Transpose so samples are rows
            cmap=cmap,
            row_cluster=cluster_samples,
            col_cluster=cluster_taxa,
            row_colors=row_colors,
            xticklabels=True,
            yticklabels=True,
            figsize=figsize
        )
        
        # Add legend
        for label, color in lut.items():
            g.ax_row_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
        g.ax_row_dendrogram.legend(title=group_var, loc="center", ncol=1)
        
        # Get the figure from the clustermap
        fig = g.fig
        
    else:
        # Plot simple heatmap
        sns.heatmap(
            abundance, 
            cmap=cmap,
            xticklabels=True,
            yticklabels=True,
            ax=ax
        )
        plt.title("Species Relative Abundance")
        plt.tight_layout()
    
    return fig


def plot_alpha_diversity_boxplot(alpha_df, metadata_df, variable, figsize=(12, 6)):
    """
    Create boxplots of alpha diversity metrics grouped by a metadata variable.
    
    Parameters:
    -----------
    alpha_df : pandas.DataFrame
        Alpha diversity DataFrame with samples as index
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    variable : str
        Metadata variable to group by
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        Boxplot figure
    """
    # Join alpha diversity with metadata
    merged = alpha_df.join(metadata_df[[variable]], how='inner')
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, len(alpha_df.columns), figsize=figsize, sharex=True)
    
    # If only one metric, convert axes to list
    if len(alpha_df.columns) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(alpha_df.columns):
        # Check if all values for this metric are the same
        if merged[metric].nunique() <= 1:
            # Just add text instead of a boxplot
            axes[i].text(0.5, 0.5, f"All values are identical\n{metric} = {merged[metric].iloc[0]:.4f}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f"{metric} Diversity")
            axes[i].set_xlabel(variable)
            # Remove axis ticks for empty plot
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        else:
            # Regular boxplot for data with variation
            try:
                sns.boxplot(x=variable, y=metric, data=merged, ax=axes[i])
                axes[i].set_title(f"{metric} Diversity")
                axes[i].set_xlabel(variable)
                
                # Rotate x-axis labels if there are many groups
                if merged[variable].nunique() > 3:
                    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            except Exception as e:
                # Handle any plotting errors
                axes[i].text(0.5, 0.5, f"Error creating plot:\n{str(e)}",
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[i].transAxes, fontsize=10, color='red')
                axes[i].set_title(f"{metric} Diversity")
                # Remove axis ticks for error plot
                axes[i].set_xticks([])
                axes[i].set_yticks([])
    
    plt.tight_layout()
    return fig

def plot_stacked_bar(abundance_df, metadata_df=None, group_var=None, top_n=10, 
                    other_category=True, sort_by=None, figsize=(12, 8)):
    """
    Create a stacked bar plot of relative abundance.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metadata_df : pandas.DataFrame, optional
        Metadata DataFrame with samples as index
    group_var : str, optional
        Metadata variable to group and sort samples by
    top_n : int, optional
        Number of most abundant species to show individually (default: 10)
    other_category : bool, optional
        Whether to include an "Other" category for remaining species (default: True)
    sort_by : str, optional
        Species to sort the samples by (default: None)
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        Stacked bar plot figure
    """
    # Copy the data to avoid modifying the original
    abundance = abundance_df.copy()
    
    # Get top N most abundant species
    mean_abundance = abundance.mean(axis=1)
    top_species = mean_abundance.nlargest(top_n).index.tolist()
    
    # Combine remaining species as "Other" if requested
    if other_category and len(abundance.index) > top_n:
        other_species = [s for s in abundance.index if s not in top_species]
        other_abundance = abundance.loc[other_species].sum()
        
        # Subset to top species
        abundance = abundance.loc[top_species]
        
        # Add "Other" row
        abundance.loc['Other'] = other_abundance
    else:
        # Subset to top species only
        abundance = abundance.loc[top_species]
    
    # Sort samples if metadata and group_var provided
    if metadata_df is not None and group_var is not None:
        # Get common samples
        common_samples = set(abundance.columns).intersection(set(metadata_df.index))
        
        # Filter to common samples
        abundance = abundance[list(common_samples)]
        meta_subset = metadata_df.loc[abundance.columns]
        
        # Sort samples by metadata group
        sample_order = meta_subset.sort_values(group_var).index
        abundance = abundance[sample_order]
    
    # Alternatively, sort by abundance of a specific species
    elif sort_by is not None and sort_by in abundance.index:
        sample_order = abundance.loc[sort_by].sort_values(ascending=False).index
        abundance = abundance[sample_order]
    
    # Transpose for plotting (species as columns)
    plot_data = abundance.T
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create stacked bar plot
    plot_data.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    
    # Customize plot
    ax.set_xlabel('Sample')
    ax.set_ylabel('Relative Abundance (%)')
    ax.set_title('Bacterial Species Composition')
    ax.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add metadata groups as background color if provided
    if metadata_df is not None and group_var is not None:
        # Get unique groups
        groups = meta_subset[group_var].unique()
        
        # Define colors
        colors = sns.color_palette("pastel", len(groups))
        
        # Create mapping
        group_mapping = {g: i for i, g in enumerate(groups)}
        
        # Get sample positions
        prev_pos = 0
        for g in groups:
            # Count samples in this group
            group_samples = meta_subset[meta_subset[group_var] == g].index
            count = sum(1 for s in abundance.columns if s in group_samples)
            
            if count > 0:
                # Add colored background
                ax.axvspan(prev_pos - 0.5, prev_pos + count - 0.5, 
                          color=colors[group_mapping[g]], alpha=0.3)
                
                # Add group label
                ax.text(prev_pos + count/2 - 0.5, 1.01, str(g), 
                       transform=ax.get_xaxis_transform(), ha='center')
                
                prev_pos += count
    
    # Rotate x-axis labels if there are many samples
    if len(abundance.columns) > 10:
        plt.xticks(rotation=90)
    
    plt.tight_layout()
    return fig


def plot_longitudinal_changes(abundance_df, metadata_df, species, time_var, subject_var, 
                             group_var=None, figsize=(12, 6)):
    """
    Create a line plot showing changes in species abundance over time.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
    species : str
        Species to plot
    time_var : str
        Metadata variable representing time point (e.g., 'Timing')
    subject_var : str
        Metadata variable identifying subjects (e.g., 'SubjectID')
    group_var : str, optional
        Metadata variable to group subjects by (e.g., 'Severity')
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        Line plot figure
    """
    # Check if species exists
    if species not in abundance_df.index:
        raise ValueError(f"Species '{species}' not found in abundance data")
    
    # Extract the species data
    species_data = abundance_df.loc[species]
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Abundance': species_data,
    })
    
    # Join with metadata
    plot_df = plot_df.join(metadata_df[[time_var, subject_var]])
    
    # Add group variable if provided
    if group_var is not None and group_var in metadata_df.columns:
        plot_df[group_var] = metadata_df.loc[plot_df.index, group_var]
    
    # Set up plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot differently based on whether group_var is provided
    if group_var is not None:
        # Plot lines for each subject, colored by group
        for subject, data in plot_df.groupby(subject_var):
            group = data[group_var].iloc[0]  # Get group for this subject
            ax.plot(data[time_var], data['Abundance'], 'o-', label=f"{subject} ({group})")
            
        # Add a mean line for each group
        for group, data in plot_df.groupby(group_var):
            means = data.groupby(time_var)['Abundance'].mean()
            ax.plot(means.index, means.values, 'k--', linewidth=2, label=f"Mean ({group})")
    else:
        # Plot lines for each subject
        for subject, data in plot_df.groupby(subject_var):
            ax.plot(data[time_var], data['Abundance'], 'o-', label=subject)
            
        # Add a mean line
        means = plot_df.groupby(time_var)['Abundance'].mean()
        ax.plot(means.index, means.values, 'k--', linewidth=2, label="Mean")
    
    # Customize plot
    ax.set_xlabel(time_var)
    ax.set_ylabel('Relative Abundance (%)')
    ax.set_title(f'{species} Abundance Over Time')
    
    # Add legend if not too many subjects
    if len(plot_df[subject_var].unique()) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_correlation_network(abundance_df, threshold=0.7, min_prevalence=0.3, 
                            method='spearman', layout='spring', figsize=(12, 12)):
    """
    Create a network plot showing correlations between species.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    threshold : float, optional
        Correlation threshold for showing connections (default: 0.7)
    min_prevalence : float, optional
        Minimum prevalence of species to include (default: 0.3)
    method : str, optional
        Correlation method ('spearman' or 'pearson')
    layout : str, optional
        Network layout algorithm ('spring', 'circular', 'random')
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        Network plot figure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("This function requires the networkx package. Install with: pip install networkx")
    
    # Filter species by prevalence
    if min_prevalence > 0:
        prevalence = (abundance_df > 0).mean(axis=1)
        abundant_species = prevalence[prevalence >= min_prevalence].index
        filtered_df = abundance_df.loc[abundant_species]
    else:
        filtered_df = abundance_df
    
    # Calculate correlation matrix
    if method == 'spearman':
        corr_matrix = filtered_df.T.corr(method='spearman')
    else:
        corr_matrix = filtered_df.T.corr(method='pearson')
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for species in corr_matrix.index:
        G.add_node(species)
    
    # Add edges for correlations above threshold
    for i, species1 in enumerate(corr_matrix.index):
        for species2 in corr_matrix.index[i+1:]:
            corr = corr_matrix.loc[species1, species2]
            if abs(corr) >= threshold:
                G.add_edge(species1, species2, weight=corr)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define node positions
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)
    
    # Define edge colors based on correlation sign
    edge_colors = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        if data['weight'] >= 0:
            edge_colors.append('green')
        else:
            edge_colors.append('red')
        edge_widths.append(abs(data['weight']) * 2)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8, node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color=edge_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Positive Correlation'),
        Line2D([0], [0], color='red', lw=2, label='Negative Correlation')
    ]
    ax.legend(handles=legend_elements)
    
    # Set title and remove axes
    ax.set_title(f'Species Correlation Network (threshold: {threshold}, method: {method})')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_abundance_summary(abundance_df, metadata_df=None, group_var=None, top_n=10):
    """
    Create a summary table of species abundance.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with species as index, samples as columns
    metadata_df : pandas.DataFrame, optional
        Metadata DataFrame with samples as index
    group_var : str, optional
        Metadata variable to group samples by
    top_n : int, optional
        Number of most abundant species to include
        
    Returns:
    --------
    pandas.DataFrame
        Summary table of species abundance
    """
    # Calculate overall mean and prevalence
    mean_abundance = abundance_df.mean(axis=1)
    prevalence = (abundance_df > 0).mean(axis=1)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Mean Abundance (%)': mean_abundance,
        'Prevalence (%)': prevalence * 100
    })
    
    # Add group-specific means if metadata provided
    if metadata_df is not None and group_var is not None:
        # Get common samples
        common_samples = set(abundance_df.columns).intersection(set(metadata_df.index))
        
        # Calculate mean abundance by group
        for group, group_df in metadata_df.loc[list(common_samples)].groupby(group_var):
            group_samples = group_df.index
            group_mean = abundance_df[group_samples].mean(axis=1)
            summary[f'Mean in {group} (%)'] = group_mean
    
    # Sort by overall mean abundance
    summary = summary.sort_values('Mean Abundance (%)', ascending=False)
    
    # Filter to top N species
    if top_n is not None and top_n < len(summary):
        summary = summary.head(top_n)
    
    return summary
