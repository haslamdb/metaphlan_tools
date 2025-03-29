# MetaPhlAn Tools

A Python package for analyzing nasal microbiome data from MetaPhlAn outputs, specifically designed for clinical studies with time-series data and clinical variables.

## Features

- Import and parse MetaPhlAn output files
- Combine multiple samples into a unified abundance table
- Filter to species-level taxonomic data
- Join with clinical metadata
- Perform diversity analyses (alpha and beta diversity)
- Identify differentially abundant species between clinical groups
- Analyze longitudinal changes in microbiome composition
- Generate publication-quality visualizations
- Command-line interface for easy usage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/metaphlan_tools.git
cd metaphlan_tools

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line interface for common analysis tasks:

#### Process MetaPhlAn Files

```bash
metaphlan_tools process --input-dir /path/to/metaphlan/files --output-dir /path/to/output
```

#### Analyze Diversity

```bash
metaphlan_tools diversity --metadata-file metadata.csv --output-dir /path/to/output --group-var Severity
```

#### Differential Abundance Analysis

```bash
metaphlan_tools differential --metadata-file metadata.csv --output-dir /path/to/output --group-var Symptoms
```

#### Longitudinal Analysis

```bash
metaphlan_tools longitudinal --metadata-file metadata.csv --output-dir /path/to/output --time-var Timing --subject-var SubjectID --group-var Severity
```

#### Generate Summary Report

```bash
metaphlan_tools report --metadata-file metadata.csv --output-dir /path/to/output --group-var Severity
```

### Python API

You can also use the package directly in your Python scripts:

```python
import pandas as pd
from metaphlan_tools import parse_metaphlan_file, combine_samples, load_metadata
from metaphlan_tools import calculate_alpha_diversity, differential_abundance_analysis
from metaphlan_tools import plot_relative_abundance_heatmap

# Process files
files = ['sample1.txt', 'sample2.txt', 'sample3.txt']
abundance_df = combine_samples(files)

# Load metadata
metadata_df = load_metadata('metadata.csv')

# Calculate alpha diversity
alpha_df = calculate_alpha_diversity(abundance_df)

# Find differentially abundant species
diff_results = differential_abundance_analysis(abundance_df, metadata_df, 'Severity')

# Create visualization
fig = plot_relative_abundance_heatmap(abundance_df, metadata_df, 'Severity')
fig.savefig('heatmap.png')
```

## Input Files

### MetaPhlAn Output Format

The package expects standard MetaPhlAn output files, which are tab-delimited with taxonomic classifications in the first column and relative abundances in subsequent columns.

### Metadata Format

The metadata should be in CSV or Excel format with:
- `SampleID` column matching the sample IDs in MetaPhlAn files
- Clinical variables including `Timing`, `Severity`, `Symptoms`, and `SubjectID`
- Additional metadata can be included and used in analyses

Example metadata structure:

```
SampleID,SubjectID,Timing,Severity,Symptoms,Age
S001,P1,Prior,0,Asymptomatic,5
S002,P1,Acute,1,Mild,5
S003,P1,Post,0,Asymptomatic,5
S004,P2,Prior,0,Asymptomatic,7
S005,P2,Acute,2,Severe,7
S006,P2,Post,0,Asymptomatic,7
```

## Example Analysis Workflow

A typical analysis workflow would consist of:

1. Process all MetaPhlAn files into a combined abundance table
2. Join with metadata for sample information
3. Calculate alpha diversity and compare between clinical groups
4. Perform beta diversity analysis to examine community-level differences
5. Identify specific species that differ between clinical groups
6. Analyze longitudinal changes across time points
7. Generate visualizations and reports

## Requirements

- Python 3.8 or higher
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-bio
- statsmodels
- scikit-learn
- networkx

## License

This project is licensed under the MIT License - see the LICENSE file for details.
