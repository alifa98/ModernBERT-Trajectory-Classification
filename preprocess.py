import pandas as pd

# Load the CSV file
csv_file = "/local/data1/shared_data/higher_order_trajectory/rome/ho_rome_res8.csv"
column_name = "higher_order_trajectory"
df = pd.read_csv(csv_file)

sequences = df[column_name].dropna().tolist()

# Save as a text file (one sequence per line) for tokenizer training
with open("sequences.txt", "w") as f:
    for seq in sequences:
        f.write(seq + "\n")
