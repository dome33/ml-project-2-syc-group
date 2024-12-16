[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Dataset

We used two existing datasets to train and test our model:
- [Handwritten Chess Scoresheet Dataset (HCS)](https://tc11.cvc.uab.es/datasets/HCS_1/) from Owen Eicher
- [Chess Reader Public Ressources](https://www.dropbox.com/scl/fo/mfoclmkggrnv0u8wufck8/h?rlkey=v0prueklq3mqsav823voin5yi&e=3&dl=0) from spinningbytes GitHub organization

In addition, we created our own dataset by handwritting publicly available chess games, and then scanning them. These data can be found in the `data/custom_dataset` folder. Each game has its separare folder denoted by `game<game_id>` which contains a `moves_san.txt` file and a scan of the handwritten game (`game<game_id>.png`).
The `moves_san.txt` files contain the moves played during the game, for both the players.

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

## Data
* The content of `2023 11 Export ChessReader Data` found in the drive provided by the professor should be placed in the `data/raw/chess_reader_dataset` folder. 
* The content of `HCS Dataset December 2021/extracted move boxes` (downloadable https://sites.google.com/view/chess-scoresheet-dataset/home/) should be placed in `datra/raw/hcs_dataset` folder. 

To generate the full training and test sets, run the following commands: 
```bash 
python src/data/extract_from_raw.py
python src/data/prepare_data.py
```

## Training. 

Each experiment (model training) should be represented by a config (`.yaml`) file in `configs` folder. 
Its results will be saved in the folder specified in the config file(usually in the `results` folder). 

To train a model, run the following command: 
```bash
python src/train/train.py --config configs/cnn_bilstm_mltu_default.yaml
```

# Dependencies 
Install mltu with 
```bash
pip install mltu
``` 

# SCITAS

## Connection to ssh

```
$ ssh USERNAME@izar.epfl.ch
```

Where USERNAME is the Gaspar account

## Uploading files from your computer to the cluster

```
rsync -azP /path/to/src <USERNAME>@izar.hpc.epfl.ch:/path/to/dest
```

Better using `rsync` that `scp` as it avoids having to re-upload a whole folder if you have only one of the files it contains that changes. This way only the changed file gets updated.

## Script 

```bash
#!/bin/bash
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=8G

module load python
python test_script.py
```

## Running a job

```
$ sbatch FILE_NAME.run
```

Where file name is the .run file as above
If the job has been successfully uploaded you should be getting a job id for it.

## Useful stuff

To see whether the state of the jobs submitted you can use the `squeue` command.
Once the job is completed there will be a new file generated with a name like SLURM-<jobid>
