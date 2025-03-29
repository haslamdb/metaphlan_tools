# example_analysis.py

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from metaphlan_tools import (
    parse_metaphlan_file, combine_samples, load_metadata, join_abundance_with_metadata,
    calculate_alpha_diversity, compare_alpha_diversity, calculate_beta_diversity, 
    perform_permanova, differential_abundance_analysis,
    plot_relative_abundance_heatmap, plot_alpha_diversity_boxplot, plot_stacked_bar, 
    plot_longitudinal_changes, plot_correlation_network, create_abundance_summary
)

# Set up directories
INPUT_DIR = "metaphlan_output"
OUTPUT_DIR = "results"
METADATA_FILE = "metadata.csv"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Process MetaPhlAn output files
print("Step 1: Processing MetaPhlAn files...")
files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
if not files:
    print(f"No files found in {INPUT_DIR}")
    exit(1)

print(f"Found {len(files)} files to process")
abundance_df = combine_samples(files)
print(f"Combined abundance table contains {len(abundance_df.index)} species across {len(abundance_df.columns)} samples")

# Save the combined abundance table
abundance_file = os.path.join(OUTPUT_DIR, "combined_abundance.csv")
abundance_df.to_csv(abundance_file)
print(f"Saved combined abundance table to {abundance_file}")

# Step 2: Load metadata
print("\nStep 2: Loading metadata...")
metadata_df = load_metadata(METADATA_FILE)
print(f"Loaded metadata for {len(metadata_df.index)} samples with {len(metadata_df.columns)} variables")

# Step 3: Calculate alpha diversity
print("\nStep 3: Calculating alpha diversity...")
alpha_df = calculate_alpha_diversity(abundance_df)
alpha_file = os.path.join(OUTPUT_DIR, "alpha_diversity.csv")
alpha_df.to_csv(alpha_file)
print(f"Saved alpha diversity metrics to {alpha_file}")

# Step 4: Compare alpha diversity between clinical groups
print("\nStep 4: Comparing alpha diversity between clinical groups...")

# Define the clinical variables to analyze
clinical_vars = ["Timing", "Severity", "Symptoms"]

for var in clinical_vars:
    if var in metadata_df.columns:
        print(f"\nAnalyzing {var}...")
        results = compare_alpha_diversity(alpha_df, metadata_df, var)
        
        # Print results
        for metric, stats in results.items():
            print(f"  {metric}: {stats['test']} p-value = {stats['p-value']:.4f}")
        
        # Create and save boxplot
        fig = plot_alpha_diversity_boxplot(alpha_df, metadata_df, var)
        boxplot_file = os.path.join(OUTPUT_DIR, f"alpha_diversity_{var}.png")
        fig.savefig(boxplot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved boxplot to {boxplot_file}")

# Step 5: Calculate beta diversity and perform PERMANOVA
print("\nStep 5: Performing beta diversity analysis...")
beta_dm = calculate_beta_diversity(abundance_df)

for var in clinical_vars:
    if var in metadata_df.columns:
        print(f"\nPerforming PERMANOVA for {var}...")
        permanova_results = perform_permanova(beta_dm, metadata_df, var)
        
        # Print results
        print(f"  PERMANOVA: p-value = {permanova_results['p-value']:.4f}, "
              f"test statistic = {permanova_results['test-statistic']:.4f}")

# Step 6: Create heatmap visualization
print("\nStep 6: Creating abundance heatmap...")
for var in clinical_vars:
    if var in metadata_df.columns:
        fig = plot_relative_abundance_heatmap(abundance_df, metadata_df, var, top_n=30)
        heatmap_file = os.path.join(OUTPUT_DIR, f"heatmap_{var}.png")
        fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved heatmap for {var} to {heatmap_file}")

# Step 7: Perform differential abundance analysis
print("\nStep 7: Identifying differentially abundant species...")
for var in clinical_vars:
    if var in metadata_df.columns:
        print(f"\nAnalyzing differential abundance for {var}...")
        da_results = differential_abundance_analysis(abundance_df, metadata_df, var)
        
        # Save results
        da_file = os.path.join(OUTPUT_DIR, f"differential_abundance_{var}.csv")
        da_results.to_csv(da_file)
        print(f"  Saved results to {da_file}")
        
        # Print top significant species
        sig_species = da_results[da_results['Adjusted P-value'] < 0.05]
        if not sig_species.empty:
            print(f"  Found {len(sig_species)} significantly different species:")
            for _, row in sig_species.head(5).iterrows():
                print(f"    {row['Species']}: adjusted p-value = {row['Adjusted P-value']:.4f}")
        else:
            print("  No significantly different species found")

# Step 8: Longitudinal analysis
print("\nStep 8: Performing longitudinal analysis...")
# Check if required variables exist in metadata
if 'Timing' in metadata_df.columns and 'SubjectID' in metadata_df.columns:
    # Get top species for analysis
    mean_abundance = abundance_df.mean(axis=1)
    top_species = mean_abundance.nlargest(5).index.tolist()
    
    for species in top_species:
        print(f"\nAnalyzing changes in {species} over time...")
        
        # Plot changes over time for each subject
        fig = plot_longitudinal_changes(
            abundance_df, 
            metadata_df, 
            species, 
            'Timing', 
            'SubjectID', 
            'Severity' if 'Severity' in metadata_df.columns else None
        )
        
        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, f"longitudinal_{species.replace(' ', '_')}.png")
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved longitudinal plot to {plot_file}")
else:
    print("Cannot perform longitudinal analysis: missing required metadata variables (Timing, SubjectID)")

# Step 9: Create correlation network
print("\nStep 9: Creating species correlation network...")
fig = plot_correlation_network(abundance_df, threshold=0.6, min_prevalence=0.3)
network_file = os.path.join(OUTPUT_DIR, "correlation_network.png")
fig.savefig(network_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved correlation network to {network_file}")

# Step 10: Create summary table
print("\nStep 10: Creating abundance summary table...")
summary_df = create_abundance_summary(abundance_df, metadata_df, 'Severity', top_n=20)
summary_file = os.path.join(OUTPUT_DIR, "abundance_summary.csv")
summary_df.to_csv(summary_file)
print(f"Saved abundance summary to {summary_file}")

print("\nAnalysis complete! Results saved to {OUTPUT_DIR}")
