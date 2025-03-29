# metaphlan_tools/cli.py

import os
import argparse
import pandas as pd
import glob
from .parser import parse_metaphlan_file, combine_samples, load_metadata, join_abundance_with_metadata
from .stats import calculate_alpha_diversity, compare_alpha_diversity, calculate_beta_diversity, perform_permanova, differential_abundance_analysis
from .viz import plot_relative_abundance_heatmap, plot_alpha_diversity_boxplot, plot_stacked_bar, plot_longitudinal_changes


def process_files(args):
    """Process MetaPhlAn output files and generate combined abundance table."""
    print(f"Processing files from {args.input_dir}...")
    
    # Find all MetaPhlAn output files
    file_pattern = os.path.join(args.input_dir, args.file_pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files.")
    
    # Combine files into a single abundance table
    abundance_df = combine_samples(files)
    
    # Save the combined table
    output_file = os.path.join(args.output_dir, "combined_abundance.csv")
    abundance_df.to_csv(output_file)
    print(f"Combined abundance table saved to {output_file}")
    
    return abundance_df


def analyze_diversity(args):
    """Analyze alpha and beta diversity."""
    # Load abundance data
    abundance_file = os.path.join(args.output_dir, "combined_abundance.csv")
    if not os.path.exists(abundance_file):
        print(f"Abundance file not found: {abundance_file}")
        print("Please run 'process' command first.")
        return
    
    abundance_df = pd.read_csv(abundance_file, index_col=0)
    
    # Load metadata
    metadata_df = load_metadata(args.metadata_file)
    
    # Calculate alpha diversity
    print("Calculating alpha diversity...")
    alpha_df = calculate_alpha_diversity(abundance_df)
    alpha_file = os.path.join(args.output_dir, "alpha_diversity.csv")
    alpha_df.to_csv(alpha_file)
    print(f"Alpha diversity saved to {alpha_file}")
    
    # Compare alpha diversity between groups
    if args.group_var:
        print(f"Comparing alpha diversity by {args.group_var}...")
        alpha_stats = compare_alpha_diversity(alpha_df, metadata_df, args.group_var)
        
        # Save results
        alpha_stats_df = pd.DataFrame(alpha_stats).T
        alpha_stats_file = os.path.join(args.output_dir, f"alpha_diversity_{args.group_var}_stats.csv")
        alpha_stats_df.to_csv(alpha_stats_file)
        print(f"Alpha diversity comparison saved to {alpha_stats_file}")
        
        # Create boxplot
        fig = plot_alpha_diversity_boxplot(alpha_df, metadata_df, args.group_var)
        boxplot_file = os.path.join(args.output_dir, f"alpha_diversity_{args.group_var}_boxplot.png")
        fig.savefig(boxplot_file, dpi=300, bbox_inches='tight')
        print(f"Alpha diversity boxplot saved to {boxplot_file}")
    
    # Calculate beta diversity
    print("Calculating beta diversity...")
    beta_dm = calculate_beta_diversity(abundance_df)
    
    # Perform PERMANOVA if group variable provided
    if args.group_var:
        print(f"Performing PERMANOVA by {args.group_var}...")
        permanova_results = perform_permanova(beta_dm, metadata_df, args.group_var)
        
        # Save results
        permanova_file = os.path.join(args.output_dir, f"permanova_{args.group_var}_results.txt")
        with open(permanova_file, 'w') as f:
            for key, value in permanova_results.items():
                f.write(f"{key}: {value}\n")
        print(f"PERMANOVA results saved to {permanova_file}")


def differential_abundance(args):
    """Perform differential abundance analysis."""
    # Load abundance data
    abundance_file = os.path.join(args.output_dir, "combined_abundance.csv")
    if not os.path.exists(abundance_file):
        print(f"Abundance file not found: {abundance_file}")
        print("Please run 'process' command first.")
        return
    
    abundance_df = pd.read_csv(abundance_file, index_col=0)
    
    # Load metadata
    metadata_df = load_metadata(args.metadata_file)
    
    # Perform differential abundance analysis
    print(f"Performing differential abundance analysis by {args.group_var}...")
    da_results = differential_abundance_analysis(abundance_df, metadata_df, args.group_var)
    
    # Save results
    da_file = os.path.join(args.output_dir, f"differential_abundance_{args.group_var}.csv")
    da_results.to_csv(da_file)
    print(f"Differential abundance results saved to {da_file}")
    
    # Create visualization for top significant species
    if not da_results.empty:
        # Get top 5 significant species
        sig_species = da_results[da_results['Adjusted P-value'] < 0.05].head(5)['Species'].tolist()
        
        for species in sig_species:
            if species in abundance_df.index:
                # Create boxplot
                fig = plot_stacked_bar(abundance_df.loc[[species]], metadata_df, args.group_var)
                plot_file = os.path.join(args.output_dir, f"species_{species.replace(' ', '_')}_{args.group_var}.png")
                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Species plot saved to {plot_file}")


def longitudinal_analysis(args):
    """Perform longitudinal analysis."""
    # Load abundance data
    abundance_file = os.path.join(args.output_dir, "combined_abundance.csv")
    if not os.path.exists(abundance_file):
        print(f"Abundance file not found: {abundance_file}")
        print("Please run 'process' command first.")
        return
    
    abundance_df = pd.read_csv(abundance_file, index_col=0)
    
    # Load metadata
    metadata_df = load_metadata(args.metadata_file)
    
    # Check required metadata variables
    required_vars = [args.time_var, args.subject_var]
    if not all(var in metadata_df.columns for var in required_vars):
        print(f"Missing required metadata variables. Need: {', '.join(required_vars)}")
        return
    
    # Get top species for longitudinal analysis
    mean_abundance = abundance_df.mean(axis=1)
    top_species = mean_abundance.nlargest(5).index.tolist()
    
    print(f"Performing longitudinal analysis for top {len(top_species)} species...")
    
    # Create longitudinal plots for each species
    for species in top_species:
        fig = plot_longitudinal_changes(
            abundance_df, 
            metadata_df, 
            species, 
            args.time_var, 
            args.subject_var, 
            args.group_var
        )
        
        plot_file = os.path.join(
            args.output_dir, 
            f"longitudinal_{species.replace(' ', '_')}_{args.time_var}.png"
        )
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Longitudinal plot for {species} saved to {plot_file}")


def create_report(args):
    """Create a summary report."""
    # Load abundance data
    abundance_file = os.path.join(args.output_dir, "combined_abundance.csv")
    if not os.path.exists(abundance_file):
        print(f"Abundance file not found: {abundance_file}")
        print("Please run 'process' command first.")
        return
    
    abundance_df = pd.read_csv(abundance_file, index_col=0)
    
    # Load metadata if provided
    metadata_df = None
    if args.metadata_file:
        metadata_df = load_metadata(args.metadata_file)
    
    # Create global abundance visualization
    print("Creating abundance heatmap...")
    fig = plot_relative_abundance_heatmap(
        abundance_df, 
        metadata_df, 
        args.group_var, 
        top_n=30
    )
    heatmap_file = os.path.join(args.output_dir, "abundance_heatmap.png")
    fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Abundance heatmap saved to {heatmap_file}")
    
    # Create stacked bar plot
    print("Creating stacked bar plot...")
    fig = plot_stacked_bar(abundance_df, metadata_df, args.group_var)
    bar_file = os.path.join(args.output_dir, "abundance_barplot.png")
    fig.savefig(bar_file, dpi=300, bbox_inches='tight')
    print(f"Stacked bar plot saved to {bar_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Nasal Microbiome Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process MetaPhlAn output files")
    process_parser.add_argument("--input-dir", required=True, help="Directory containing MetaPhlAn files")
    process_parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    process_parser.add_argument("--file-pattern", default="*.txt", help="Pattern to match MetaPhlAn files")
    
    # Diversity command
    diversity_parser = subparsers.add_parser("diversity", help="Analyze microbial diversity")
    diversity_parser.add_argument("--metadata-file", required=True, help="Path to metadata file")
    diversity_parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    diversity_parser.add_argument("--group-var", help="Metadata variable to group by")
    
    # Differential abundance command
    diff_parser = subparsers.add_parser("differential", help="Perform differential abundance analysis")
    diff_parser.add_argument("--metadata-file", required=True, help="Path to metadata file")
    diff_parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    diff_parser.add_argument("--group-var", required=True, help="Metadata variable to group by")
    
    # Longitudinal command
    time_parser = subparsers.add_parser("longitudinal", help="Perform longitudinal analysis")
    time_parser.add_argument("--metadata-file", required=True, help="Path to metadata file")
    time_parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    time_parser.add_argument("--time-var", required=True, help="Metadata variable for time point")
    time_parser.add_argument("--subject-var", required=True, help="Metadata variable for subject ID")
    time_parser.add_argument("--group-var", help="Metadata variable to group by")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate summary report")
    report_parser.add_argument("--metadata-file", help="Path to metadata file")
    report_parser.add_argument("--output-dir", required=True, help="Directory to save output files")
    report_parser.add_argument("--group-var", help="Metadata variable to group by")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if hasattr(args, 'output_dir') and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute command
    if args.command == "process":
        process_files(args)
    elif args.command == "diversity":
        analyze_diversity(args)
    elif args.command == "differential":
        differential_abundance(args)
    elif args.command == "longitudinal":
        longitudinal_analysis(args)
    elif args.command == "report":
        create_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
