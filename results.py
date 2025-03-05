import pandas as pd
import os

# Directory containing .txt files
results_dir = "results_techniques"

# List all .txt files in the directory
files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]

# Initialize a list to hold data for DataFrame
data = []

# Process each file
for file in files:
    file_path = os.path.join(results_dir, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Strip whitespace and newline characters
    values = [line.strip() for line in lines]
    
    # Create a dictionary with the file name and values as columns
    metrics = {
        'File': file
    }
    
    # Add values to the dictionary with dynamic column names
    for idx, value in enumerate(values):
        metrics[f'Value_{idx + 1}'] = value
    
    data.append(metrics)

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_excel_path = 'results.xlsx'
df.to_excel(output_excel_path, index=False, engine='openpyxl')

print(f"Metrics and values have been saved to {output_excel_path}")
