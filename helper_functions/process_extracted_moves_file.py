def process_moves_file(input_file, output_file):
    """
    Reads a .txt file line by line, processes each line to reformat the filenames,
    and writes the updated lines to a new .txt file.

    Args:
        input_file (str): Path to the input .txt file.
        output_file (str): Path to the output .txt file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                # Split the line into filename and move
                filename, move = line.strip().split(" ", 1)
                
                # Extract parts of the filename
                parts = filename.split("_")
                if len(parts) == 3:  # Ensure the filename has the expected format
                    game_id = parts[0]        # First part is game ID
                    # page_number = parts[1]    # Second part is ignored (page number)
                    move_number = parts[1]    # Third part is move number
                    colour = parts[2].split(".")[0]  # Extract colour and remove '.png'
                    
                    # Create the new filename
                    new_filename = f"dataset1_{game_id}_{colour}_{move_number}.png"
                    
                    # Write the updated line to the output file
                    outfile.write(f"{new_filename} {move}\n")
                    print(f"Processed: {filename} -> {new_filename}")
                else:
                    print(f"Skipped: {line.strip()} (unexpected format)")
            except Exception as e:
                print(f"Error processing line: {line.strip()} - {e}")