import os
import pandas as pd
import re  # For regex operations


# Function to extract iteration number from filename
def extract_iteration_number(filename):
    match = re.search(r"iter(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# User-provided lists
pnrs = []  # Initially empty; to process all unique values if empty
columns_of_interest = ["LotDemand", "SP", "util"]  # Columns you're interested in

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize an empty list to store data from each CSV
dataframes = []
all_lots = set()  # To store all unique Lot values if pnrs is empty

# Loop through all CSV files in the current directory
for filename in os.listdir(current_dir):
    if filename.endswith(".csv"):
        # Construct full path to the CSV file
        file_path = os.path.join(current_dir, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Update all_lots with unique values from this CSV if pnrs is empty
        if not pnrs:
            all_lots.update(df["Lot"].unique())

        # Append the DataFrame and filename as a tuple
        dataframes.append((df, filename))

# If pnrs was empty, use all unique Lot values found
if not pnrs:
    pnrs = list(all_lots)

# Process each DataFrame to filter and extract necessary data
filtered_dataframes = []
for df, source_filename in dataframes:
    # Filter rows where 'Lot' matches values in the pnrs list
    filtered_df = df[df["Lot"].isin(pnrs)]

    # Extract columns of interest, ensure 'Lot' is included for reference, and round the data
    filtered_df = filtered_df[["Lot"] + columns_of_interest].round(2)
    filtered_df["SourceFile"] = source_filename
    # Extract iteration number and add as a new column
    filtered_df["IterNumber"] = extract_iteration_number(source_filename)

    # Append the filtered DataFrame to the list
    filtered_dataframes.append(filtered_df)

# Combine all filtered DataFrames into one
combined_df = pd.concat(filtered_dataframes, ignore_index=True)

# Sort by 'Lot', 'IterNumber', then 'SourceFile'
combined_df.sort_values(by=["Lot", "IterNumber", "SourceFile"], inplace=True)

# Drop the 'IterNumber' column if not needed in the final output
combined_df.drop("IterNumber", axis=1, inplace=True)

# Path for the output Excel file in the same directory as the script
output_excel_path = os.path.join(current_dir, "comparison.xlsx")

# Write the combined and sorted DataFrame to an Excel file
combined_df.to_excel(output_excel_path, index=False)

print(f"Data extracted and saved to {output_excel_path}")
