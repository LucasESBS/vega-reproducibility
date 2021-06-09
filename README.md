# VEGA reproducibility


### IMPORTANT NOTE (2021/06/04): 
**Reproducibility code updated for current revisions is available in "revision1" branch. Please clone this specific branch for up-to-date reproducibility code.**

This repository contains the code and instructions to reproduce the results from the VEGA paper.

Set up your own environment with Conda and install required dependencies:
```
pip install -r requirements.txt
```
Instructions:

1) Download archive with preprocessed data at:
https://drive.google.com/file/d/1j0D6gMQEGmwSfS7GeO_ZB4TsyGh0eN5y/view?usp=sharing

2) Unpack data to the data directory using this command in this directory:
```
tar zxvf /path/to/vega-data-rev1.tar.gz -C ./data --strip-components 1
```
3) Clone the dev branch of VEGA package in this directory
```
git clone -b vega_dev https://github.com/LucasESBS/vega.git
```
4) Run the bash script to train all models:
```
bash train_all_models.sh
```
Once done training, models are in ./trained_models/

5) Use the jupyter notebooks to reproduce figures
