import csv

def process_csv_to_txt(input_csv, output_txt):
    """
    Script to extract the data in the .csv file and write it to a .txt file in the required format.

    Args:
        input_csv (str): Path to the input .csv file.
        output_txt (str): Path to the output .txt file.
    """
    with open(input_csv, 'r') as csv_file, open(output_txt, 'w') as txt_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            try:
                # Extract the 'corrected' and 'box' values
                corrected = row.get('corrected', '').strip()
                box = row.get('box', '').strip()
                
                if corrected and box:  # Ensure both columns have data
                    # Extract parts from the 'box' column
                    parts = box.split("_")
                    if len(parts) >= 5:  # Validate the format of the 'box' column
                        game_id = parts[1]  # Second part is the game ID
                        page_number = parts[3]  # Fourth part (unrelevant numbers)
                        move_number = parts[4][1:3]  # Extract move number from last part
                        colour = "black" if parts[4][0] == "b" else "white"  # Determine the colour
                        
                        # Create the formatted filename
                        formatted_name = f"dataset2_{game_id}_{page_number}_{move_number}_{colour}.png"
                        
                        # Write to the output file
                        txt_file.write(f"{formatted_name} {corrected}\n")
                        print(f"Processed: {box} -> {formatted_name} {corrected}")
                    else:
                        print(f"Skipped invalid 'box' format: {box}")
                else:
                    print(f"Skipped row with missing 'corrected' or 'box': {row}")
            except Exception as e:
                print(f"Error processing row: {row} - {e}")

