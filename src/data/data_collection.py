import cv2
import numpy as np
import matplotlib.pyplot as plt
import chess.pgn 
import argparse 
import os 


def get_box_positions(img)-> list[tuple[int, int, int, int]]: 
    """
    Returns the positions of the boxes in the image (x, y, width, height) 
    Hardcoded based on the CHESS.COM scoresheet layout. 
    
    :param img: np.ndarray, the image
    :return: list of tuples, the positions of the boxes in the image
    """
    
    # Values are hardcoded based on the CHESS.COM scoresheet layout
    x , y , w , h = 320, 628, 363, 80
    
    rects = [] 
    # We extract the first 25 boxes for the white player
    for i in range(0, 25): 
        new_x = x 
        new_y = y + i * h
        rects.append((new_x, new_y, w, h))
        
    # We extract the first 25 boxes for the black player
    for i in range(0, 25): 
        new_x = x + w 
        new_y = y + i * h
        rects.append((new_x, new_y, w, h))
        
    # Sort the rects by y and then x
    rects = sorted(rects, key=lambda x: (x[1],x[0])) 

    right_rects = [] 
    # We extract the next 25 boxes for the white player
    for i in range(0, 25): 
        new_x = x + 2 * w + 100  
        new_y = y + i * h
        right_rects.append((new_x, new_y, w, h))

    # We extract the next 25 boxes for the black player
    for i in range(0, 25): 
        new_x = x + 3 * w + 100
        new_y = y + i * h
        right_rects.append((new_x, new_y, w, h))
    right_rects = sorted(right_rects, key=lambda x: (x[1],x[0])) 

    rects.extend(right_rects) 
    return rects 


def extract_moves_from_image(img_path:str)->list[np.ndarray]: 
    """
    Extracts the moves from the image at the given path. 
    
    :param img_path: str, the path to the image
    :return: list of np.ndarrays, the extracted moves
    """
    img = cv2.imread(img_path) 

    # Get the positions of the boxes in the image
    rects = get_box_positions(img) 

    moves = [] 
    # Extract the box from the image
    for i, rect in enumerate(rects): 
        x, y, w, h = rect 
        crop = img[y:y+h, x:x+w]
        moves.append(crop) 
    return moves


def process_chess_dataset(dataset_path: str, max_folders: int = None, offset: int = 50): 
    """
    Processes the custom dataset, extracts box images from PNG files, and maps them to chess moves.
    Saves the result in a single box-to-move mapping file located in the dataset directory.
    
    :param dataset_path: str, the path to the 'custom_dataset' folder
    :param max_folders: int, the maximum number of data folders to process. Processes all if None.
    """

    # Get all game folders in the dataset
    game_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    game_folders = sorted(game_folders)  # Ensure consistent processing order
    
    if max_folders is not None:
        game_folders = game_folders[:max_folders]

    # Create a list to hold all box-to-move mappings
    all_box_to_move = []

    for game_folder in game_folders:
        print(f"Processing {game_folder}...")
        game_folder_path = os.path.join(dataset_path, game_folder)
        
        # Find all PNG files in the folder
        png_files = [os.path.join(game_folder_path, f) for f in os.listdir(game_folder_path) if f.startswith("game")]
        moves_file_path = os.path.join(game_folder_path, "moves_san.txt")
        
        if not png_files or not os.path.exists(moves_file_path):
            print(f"Missing required files in {game_folder_path}. Skipping...")
            continue
        
        # Read moves from the moves_san.txt file
        try:
            with open(moves_file_path, "r") as f:
                moves_lines = f.readlines()
        except Exception as e:
            print(f"Error reading moves file in {game_folder_path}: {e}")
            continue
        
        # Parse moves
        moves = []
        for line in moves_lines:
            parts = line.strip().split()
            if len(parts) == 3:
                _, white_move, black_move = parts
                moves.append(white_move)
                moves.append(black_move)
        
        # Extract boxes from each PNG and save cropped images
        move_index = 0
        for png_file in png_files:
            img = cv2.imread(png_file)
            box_positions = get_box_positions(img)
            for rect in box_positions:
                if move_index >= len(moves):
                    break
                
                x, y, w, h = rect
                
                box_img = img[y:y+h, x:x+w]
                box_img = img[y-offset:y+h+offset, x-offset:x+w+offset]
                
                relatitve_box_img_path = os.path.join(game_folder, f"box_{move_index}.png")
                box_img_path = os.path.join(dataset_path, relatitve_box_img_path) 
                cv2.imwrite(box_img_path, box_img)
                
                # Add the box-to-move mapping to the list in the required format
                all_box_to_move.append(f"{relatitve_box_img_path},{moves[move_index]}")
                move_index += 1

    # Write all box-to-move mappings to a single text file in the dataset directory
    output_txt_path = os.path.join(dataset_path, "labels.txt")
    try:
        with open(output_txt_path, "w") as f:
            # Write the header first
            f.write("id,prediction\n")
            # Write the box-to-move mappings
            f.writelines("\n".join(all_box_to_move))
        print(f"All box-to-move mappings saved to: {output_txt_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")

   
def extract_game_moves_san_frompgn(game_idx, destination_path, pgns_dataset_path):
    """
    Extract the moves of a specified game from a PGN file and save them in SAN notation.
    
    :param game_idx: int, the index of the game in the PGN file
    :param destination_path: str, the directory to save the moves
    :param pgns_dataset_path: str, the path to the PGN file
    """
    
    with open(pgns_dataset_path) as pgn_file:
        # Locate the desired game
        for _ in range(int(game_idx) + 1):  # Skip to the game index
            game = chess.pgn.read_game(pgn_file)
        
        if not game:
            print(f"Game {game_idx} not found.")
            return
        
        # Get moves in SAN notation
        board = game.board()
        moves_san = []
        for move in game.mainline_moves():
            moves_san.append(board.san(move))
            board.push(move)
        
        # Create directory for saving moves
        dir_path = os.path.join(destination_path, f"game{game_idx}")
        os.makedirs(dir_path, exist_ok=True)

        # Write SAN moves to a file
        san_str = ""
        for i, move in enumerate(moves_san):
            if i % 2 == 0:
                san_str += f"\n{i // 2 + 1}."
            san_str += f" {move}"
        
        with open(os.path.join(dir_path, "moves_san.txt"), "w") as f:
            f.write(san_str.strip())
    
    print(f"SAN moves saved for game {game_idx} in {dir_path}")
            


if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--process", action="store_true") 
    parser.add_argument("--game_id", type=str, default=None) 
    parser.add_argument("--destination_path", default=None) 
    parser.add_argument("--pgns_dataset_path", type=str, default=None) 
    args = parser.parse_args()
    
    if args.extract: 
        assert args.game_id is not None
        assert args.destination_path is not None
        assert args.pgns_dataset_path is not None
        extract_game_moves_san_frompgn(args.game_id, args.destination_path, args.pgns_dataset_path)
    if args.process:
        assert args.destination_path is not None 
        process_chess_dataset(args.destination_path)
