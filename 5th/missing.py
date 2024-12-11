import pandas as pd
import numpy as np

def analyze_missing_values(df, print_samples=True):
    """
    Analyze missing values in a DataFrame and print summary statistics
    only for columns that have missing values.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame to analyze
    print_samples (bool): If True, print sample values for columns with missing data
    
    Returns:
    pandas.DataFrame: Summary of missing values statistics
    """
    # Get total missing values per column
    missing_counts = df.isnull().sum()
    
    # Filter only columns with missing values
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        print("No missing values found in the DataFrame!")
        return None
    
    # Calculate statistics for columns with missing values
    stats = []
    for col in missing_cols.index:
        # Calculate basic statistics
        total_missing = missing_counts[col]
        percent_missing = (total_missing / len(df)) * 100
        
        # Get data type and unique values count (excluding NA)
        dtype = df[col].dtype
        unique_count = df[col].dropna().nunique()
        
        # Prepare statistics dictionary
        col_stats = {
            'Column': col,
            'Missing Count': total_missing,
            'Missing %': round(percent_missing, 2),
            'Data Type': dtype,
            'Unique Values': unique_count
        }
        
        # Print detailed information
        print(f"\nColumn: {col}")
        print(f"Missing Values: {total_missing:,} ({percent_missing:.2f}%)")
        print(f"Data Type: {dtype}")
        print(f"Unique Values (excluding NA): {unique_count:,}")
        
        if print_samples and total_missing < len(df):
            # Print some sample values where data exists
            print("\nSample of existing values:")
            sample_values = df[col].dropna().sample(min(5, unique_count)).values
            print(sample_values)
            
            # If numeric, print basic statistics
            if np.issubdtype(dtype, np.number):
                print("\nNumeric Statistics:")
                print(f"Mean: {df[col].mean():.2f}")
                print(f"Median: {df[col].median():.2f}")
                print(f"Std Dev: {df[col].std():.2f}")
        
        stats.append(col_stats)
    
    # Create and return summary DataFrame
    summary_df = pd.DataFrame(stats)
    summary_df = summary_df.set_index('Column')
    
    return summary_df

# Example usage:
if __name__ == "__main__":
    # Load your DataFrame here
    df = pd.read_csv('encoded_departs.csv')
    df = df.drop(columns=['CONSPUMA', 'CPUMA0010','APPAL','APPALD'])
        
    # Analyze missing values
    summary = analyze_missing_values(df)
    
    # Print overall summary
    print("\nOverall Summary:")
    print(summary)