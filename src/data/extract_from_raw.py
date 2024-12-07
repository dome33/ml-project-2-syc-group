import os
import shutil
import pandas as pd

"""
The python files in this folder are the one used to create a unified dataset

The files with the boxes images get renamed to match the following format: `<dataset>_<game id>_<colour>_<move number>.png` where:
- `<dataset>` is either 'dataset1' or 'dataset2', depending on which initial dataset the image came from.
- `<game_id>` is the game number, it enable us to know which picture go together.
- `<colour>` is either 'black' or 'white' depending on which player played the given move.
- `<move_number>` is the id of the move. It enable us to know the chronology of the game.

The tags.txt file has been created using the two helper functions contained in `process_box_file.py` and `process_extracted_moves_file.py`. 
It contains one line for each of the files in the following format: `<file name> <corresponding move>` where:
- `<file name>` is in the same format as specified above.
- `<corresponding move>` is the label of the file.

"""

def process_online_dataset(txt_file_path, source_folder, destination_folder, output_txt_path): 
    """
    Processes a text file where each line has the format `<file name> <corresponding move>`,
    copies the file from the source folder to the destination folder, and renames it by
    prepending 'onlinedata_' to the file name.

    Args:
        txt_file_path (str): Path to the text file with file names and moves.
        source_folder (str): Path to the source folder containing the files.
        destination_folder (str): Path to the destination folder to copy the files to.
        output_txt_path (str): Path to the .txt file to be created. 
    Returns:
        None
    """
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    with open(txt_file_path, 'r') as file , open(output_txt_path, 'w') as output_file: 
        output_file.write("id,prediction\n") 
        
        for line in file:
            # Extract the file name from the line
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 1:
                print(f"Skipped empty or malformed line: {line}")
                continue
            # Write the file name and move to the output text file 
            # img_id = parts[0].split(".")[0] 
            output_file.write(f"{parts[0]},{parts[1]}\n")
            
            original_file_name = parts[0]
            # Construct source and destination paths
            source_path = os.path.join(source_folder, original_file_name)
            new_file_name = f"{original_file_name}"
            destination_path = os.path.join(destination_folder, new_file_name)

            if os.path.exists(source_path):
                # Copy and rename the file
                shutil.copy(source_path, destination_path)
                print(f"Copied and renamed: {original_file_name} -> {new_file_name}")
            else:
                print(f"File not found: {source_path}")
            
            
            


def process_prof_dataset(csv_path, source_folder, destination_folder, output_txt_path, filter_path):
    """
    Processes a CSV file and copies corresponding files to a new folder with renamed files.
    Additionally, creates a .txt file logging the renamed files and their corresponding moves.

    Args:
        csv_path (str): Path to the CSV file.
        source_folder (str): Path to the folder containing the source files.
        destination_folder (str): Path to the folder where files will be copied and renamed.
        output_txt_path (str): Path to the .txt file to be created.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows where 'move_state' is 'valid'
    valid_moves = df[df['move_state'] == 'valid']
    
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # read from filter_path 
    with open(filter_path, 'r') as f:
        filtered_names = f.read().splitlines()
        filtered_names = set(filtered_names)
        
    # Open the output text file for writing
    with open(output_txt_path, 'w') as output_file:
        output_file.write("id,prediction\n") 
    
        for _, row in valid_moves.iterrows():
            # Determine the original file name using the 'id' column
            file_id = int(row['id'])
            png_filename = f"{file_id}.png"
            jpe_filename = f"{file_id}.jpe"
            
            if png_filename in filtered_names : 
                print(f"Skipped {png_filename}")
                continue 
            # Check if the file exists as .png or .jpe
            if os.path.exists(os.path.join(source_folder, png_filename)):
                original_filename = png_filename
            elif os.path.exists(os.path.join(source_folder, jpe_filename)):
                original_filename = jpe_filename
            else:
                print(f"File not found for ID {file_id}: neither {png_filename} nor {jpe_filename}")
                raise FileNotFoundError(f"File not found for ID {file_id}: neither {png_filename} nor {jpe_filename}") 
            
            # Full path of the original file
            original_filepath = os.path.join(source_folder, original_filename)
            
            # Create the new file name and its path
            new_filename = f"{png_filename}"
            new_filepath = os.path.join(destination_folder, new_filename)
            
            # Copy the file to the destination with the new name
            shutil.copy(original_filepath, new_filepath)
            
            # Write the new file name and the corresponding move to the .txt file
            corresponding_move = row['prediction']
            output_file.write(f"{png_filename},{corresponding_move}\n")
            
            print(f"Processed: {file_id} -> {new_filename}")
            

            
if __name__ == "__main__":
    # Example usage
    txt_file_path = "data/raw/hcs_dataset/extracted move boxes/train_data.txt" 
    source_folder = "data/raw/hcs_dataset/extracted move boxes"
    destination_folder = "data/online_dataset"
    output_txt_path = "data/online_dataset/labels.txt" 
    process_online_dataset(txt_file_path, source_folder, destination_folder, output_txt_path) 

    csv_path = "data/raw/chess_reader_data/prediciton.csv"
    source_folder = "data/raw/chess_reader_data/images" 
    destination_folder = "data/chess_reader_dataset"
    output_txt_path = "data/chess_reader_dataset/labels.txt" 
    filter_path = "data/chess_reader_filter.txt"
    process_prof_dataset(csv_path, source_folder, destination_folder, output_txt_path, filter_path)