# metaphlan_tools/parser.py

import os
import pandas as pd
import numpy as np
import re


def parse_metaphlan_file(file_path, sample_id=None):
    """
    Parse a MetaPhlAn output file and extract species-level information.
    
    Parameters:
    -----------
    file_path : str
        Path to the MetaPhlAn output file
    sample_id : str, optional
        Sample ID to associate with this data. If None, will try to extract from filename
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing species-level relative abundances
    """
    # If sample_id is not provided, try to extract from filename
    if sample_id is None:
        sample_id = os.path.basename(file_path).split('.')[0]
    
    # Read the MetaPhlAn output file
    try:
        df = pd.read_csv(file_path, sep='\t', header=0, comment='#')
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty file or parsing error for {file_path}")
        return pd.DataFrame()
    
    # Get the column containing taxonomic information (usually the first column)
    if df.shape[1] < 2:
        print(f"Warning: File format error in {file_path}")
        return pd.DataFrame()
    
    # Assume first column contains taxonomic info, second column contains abundance
    tax_col = df.columns[0]
    abund_col = df.columns[1]
    
    # Filter to only include species level entries
    species_df = df[df[tax_col].str.contains('s__[^|]', regex=True)]
    
    # Extract species names
    species_df['Species'] = species_df[tax_col].apply(
        lambda x: re.search(r's__([^|]+)(?:\|t__[^|]+)?$', x).group(1) if re.search(r's__([^|]+)(?:\|t__[^|]+)?$', x) else None
    )
    
    # Create a new DataFrame with species as index and sample as column
    result_df = pd.DataFrame({
        'Species': species_df['Species'],
        sample_id: species_df[abund_col]
    }).set_index('Species')
    
    return result_df


def combine_samples(file_paths, sample_ids=None):
    """
    Parse multiple MetaPhlAn output files and combine them into a single DataFrame.
    
    Parameters:
    -----------
    file_paths : list
        List of paths to MetaPhlAn output files
    sample_ids : list, optional
        List of sample IDs to associate with each file
        
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame containing species-level relative abundances for all samples
    """
    # If sample_ids not provided, extract from filenames
    if sample_ids is None:
        sample_ids = [os.path.basename(fp).split('.')[0] for fp in file_paths]
    
    # Check that the lengths match
    if len(file_paths) != len(sample_ids):
        raise ValueError("Number of file paths and sample IDs must match")
    
    # Parse each file
    dfs = []
    for fp, sid in zip(file_paths, sample_ids):
        df = parse_metaphlan_file(fp, sid)
        if not df.empty:
            dfs.append(df)
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, axis=1)
        # Fill NaN values with 0 (species not detected in a sample)
        combined_df = combined_df.fillna(0)
        return combined_df
    else:
        return pd.DataFrame()


def load_metadata(metadata_file, sample_id_col='SampleID'):
    """
    Load sample metadata from a CSV or Excel file.
    
    Parameters:
    -----------
    metadata_file : str
        Path to the metadata file
    sample_id_col : str, optional
        Name of the column containing sample IDs
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metadata with sample ID as index
    """
    # Determine file type based on extension
    if metadata_file.endswith('.csv'):
        metadata = pd.read_csv(metadata_file)
    elif metadata_file.endswith(('.xlsx', '.xls')):
        metadata = pd.read_excel(metadata_file)
    else:
        raise ValueError("Metadata file must be CSV or Excel format")
    
    # Set sample ID as index
    if sample_id_col in metadata.columns:
        metadata = metadata.set_index(sample_id_col)
    else:
        raise ValueError(f"Sample ID column '{sample_id_col}' not found in metadata")
    
    return metadata


def join_abundance_with_metadata(abundance_df, metadata_df):
    """
    Join the abundance table with metadata.
    
    Parameters:
    -----------
    abundance_df : pandas.DataFrame
        Species abundance DataFrame with samples as columns
    metadata_df : pandas.DataFrame
        Metadata DataFrame with samples as index
        
    Returns:
    --------
    pandas.DataFrame
        Transposed abundance DataFrame with metadata columns added
    """
    # Transpose abundance table to have samples as rows
    abundance_transposed = abundance_df.T
    
    # Check for sample ID overlap
    common_samples = set(abundance_transposed.index).intersection(set(metadata_df.index))
    if not common_samples:
        raise ValueError("No common sample IDs found between abundance data and metadata")
    
    # Join with metadata
    joined_df = abundance_transposed.join(metadata_df, how='inner')
    
    return joined_df
