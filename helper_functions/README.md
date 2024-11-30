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
