import os

def remap_labels(directory):
    """
    Goes through all .txt files in a directory, reads each line,
    changes the class_id (the first number) to 0, and rewrites the file.
    """
    print(f"Processing directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            lines_to_write = []
            with open(filepath, 'r') as f_read:
                for line in f_read.readlines():
                    parts = line.strip().split()
                    # Change the first part (class_id) to '0'
                    parts[0] = '0'
                    lines_to_write.append(" ".join(parts))
            
            # Write the modified lines back to the same file
            with open(filepath, 'w') as f_write:
                for line in lines_to_write:
                    f_write.write(line + '\n')
    print("Done.")

# --- IMPORTANT ---
# Define the paths to your label folders
base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset'))
train_labels_path = os.path.join(base_data_path, 'train', 'labels')
valid_labels_path = os.path.join(base_data_path, 'valid', 'labels')
test_labels_path = os.path.join(base_data_path, 'test', 'labels')

# Run the remapping function on all your dataset splits
remap_labels(train_labels_path)
remap_labels(valid_labels_path)
remap_labels(test_labels_path)