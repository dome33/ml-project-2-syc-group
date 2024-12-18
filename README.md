[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Machine Learning for Chess Movement Recognition

## Datasets

We used two existing datasets to train and test our model:
- [Handwritten Chess Scoresheet Dataset (HCS)](https://tc11.cvc.uab.es/datasets/HCS_1/) from Owen Eicher
- [Chess Reader Public Ressources](https://www.dropbox.com/scl/fo/mfoclmkggrnv0u8wufck8/h?rlkey=v0prueklq3mqsav823voin5yi&e=3&dl=0) from spinningbytes GitHub organization

In addition, we created our own dataset by handwritting publicly available chess games, and then scanning them. These data can be found in the `data/custom_dataset` folder. Each game has its separare folder denoted by `game<game_id>` with the following structure:

```
├── game<game_id>
│   ├── game<game_id>.png     # Scan of the handwritten game                  
│   ├── moves_san.txt         # The moves played during the game
```

## Downloading the data

* The content of `2023 11 Export ChessReader Data` found in the drive provided by the professor should be placed in the `data/raw/chess_reader_data` folder. 
* The content of `HCS Dataset December 2021/extracted move boxes` (downloadable https://sites.google.com/view/chess-scoresheet-dataset/home/) should be placed in `data/raw/hcs_dataset` folder.

More informations on the scripts to process the data can be found in the [dedicated README.md](https://github.com/CS-433/ml-project-2-syc-group/blob/main/src/data/README.md)

To generate the full training and test sets, run the following commands from the root of the project: 
```bash
python src/data/existing_datasets.py
python src/data/custom_dataset.py --process --destination_path ./data/custom_dataset
python src/data/prepare_data.py
```

## Project structure 

```
├── configs                   # Configuration for the models
├── data/                     # Unprocessed datasets
│   ├── custom_dataset/       # Scanned sheets and corresponding move 
├── notebooks/                
├── results/                  
├── src/                      # Source code 
│   ├── data/                 # Scripts for loading and preprocessing datasets
│   ├── models/               
│   ├── train/                # Training and evaluation scripts
├── .env                      
├── README.md
├── requirements.txt
├── run.sh            
```

## Training

Each experiment (model training) should be represented by a config (`.yaml`) file in `configs` folder. 
The results will be saved in the folder specified in the config file (usually in the `results` folder). 

To train a model, run the following command: 
```bash
python src/train/train.py --config configs/cnn_bilstm_mltu_default.yaml
```

## Dependencies 

Install the requirements using the following command:

```bash
pip install -r requirements.txt
``` 
