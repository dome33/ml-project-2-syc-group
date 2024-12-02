import os

def rename_dataset1_files(folder_path):
    """
    Renames files in a folder that start with 'dataset1' by removing the first '.png' 
    in their names. Other files are not renamed.

    Args:
        folder_path (str): Path to the folder containing the files to rename.
    """
    for filename in os.listdir(folder_path):
        # Process only files starting with 'dataset1' and containing '.png_'
        if filename.startswith("dataset1") and ".png_" in filename:
            # Construct the new filename
            new_filename = filename.replace(".png_", "_", 1)  # Replace the first occurrence only
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        else:
            print(f"Skipped: {filename}")

# Example usage
# Replace with the path to your folder
folder_path = "dataset/merged_dataset/data"
rename_dataset1_files(folder_path)
