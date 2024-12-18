The python files in this folder are the ones used to create a unified dataset

## custom_dataset.py

This file contains scripts related to the custom dataset we created.

### Standard Algebraic Notation (SAN) extraction
Extracts the moves of a specified game from a PGN file and save them in SAN notation.

**Example Usage:**
```
python custom_dataset.py --extract --game_id 1 --destination_path ./output --pgns_dataset_path ./games.pgn
```

### Process the dataset

Processes the custom dataset by extracts box images from PNG files, and mapping them to chess moves.
Saves the result in a single box-to-move mapping file located in the dataset directory.

**Example Usage:**
```
python custom_datset.py --process --destination_path ./custom_dataset
```


## existing_datasets.py

This file contains scripts to process both the [Handwritten Chess Scoresheet (HCS)](https://tc11.cvc.uab.es/datasets/HCS_1/) dataset and the ChessReader dataset.

The files used as input and output are specified in the python script and can easily be changed. It produces a `labels.txt` file containing image-to-move mappings in CSV format (id,prediction) and it organizes image files in their respective folders.

**Example Usage:**
```
python existing_datasets.py
```


## prepare_data.py

This file contains a script to prepare the three datasets for training, validation and testing.
It loads image paths and labels from three datasets, splits the datasets and shuffles them before saving the splits.
It takes an optional argument `--size_custom` to allow a limit of the size of the custom dataset.

**Example Usage:**
```
python prepare_data.py --size_custom 10
```
