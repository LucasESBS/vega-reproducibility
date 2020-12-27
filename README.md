VEGA reproducibility

This repository contains the code and instructions to reproduce the results from the VEGA paper.

Set up your own environment with Conda and install required dependencies:
pip install -r requirements.txt

Instructions:

1) Download archive with preprocessed data at:
https://drive.google.com/file/d/17suAzOKkiLwdv-IVPuJgPbhs45cKfT0x/view?usp=sharing

2) Unpack data to the data directory using this command in this directory:
tar zxvf /path/to/vega-data.tar.gz -C ./data --strip-components 1

3) Run the bash script to train all models:
bash train_all_models.sh

Once done training, models are in ./trained_models/

4) Use the jupyter notebooks to reproduce figures
