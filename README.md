[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Dataset

The unified dataset can be found in this [Google drive folder](https://drive.google.com/drive/u/1/folders/1RY_eGbldIL4Ig_2OHbMjVMlfOV_o27r_)
The two datasets used to create this unified dataset are:
- [Handwritten Chess Scoresheet Dataset (HCS)](https://tc11.cvc.uab.es/datasets/HCS_1/) from Owen Eicher
- [Chess Reader Public Ressources](https://www.dropbox.com/scl/fo/mfoclmkggrnv0u8wufck8/h?rlkey=v0prueklq3mqsav823voin5yi&e=3&dl=0) from spinningbytes GitHub organization

# SCITAS

## Connection to ssh

```
$ ssh USERNAME@izar.epfl.ch
```

Where USERNAME is the Gaspar account

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

## Uploading a file to the ssh machine

```
$ scp path/to/file USERNAME@izar.hpc.epfl.ch:path/to/file
```
