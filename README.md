[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

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
