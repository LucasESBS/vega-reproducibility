echo "Creating subdirectories"
mkdir -p paper/figures/
echo "Training model for Kang PBMC..."
mkdir -p trained_models/kang_pbmc/ 
python src/train_vega_kang_pbmc.py
echo "DONE"
echo "Training model for SuppFig1..."
mkdir -p trained_models/compare_models/
python src/train_vega_suppFig1.py
python src/train_sparse_vae_suppFig1.py
python src/train_deep_decoder_suppFig1.py
python src/train_vanilla_vae_suppFig1.py
echo "DONE"
echo "Training model for MIX-Seq..."
mkdir -p trained_models/mixseq/
python src/train_vega_mixseq.py
echo "DONE"
echo "Training model for GBM..."
mkdir -p trained_models/gbm/
python src/train_vega_gbm.py
echo "DONE"
echo "Training model for organoid..."
mkdir -p trained_models/organoid/
python src/train_vega_organoid.py
echo "DONE"
echo "Training model for Fig4."
mkdir -p trained_models/kang_ctout_models/
python src/train_vega_ct_out_pbmc.py
mkdir -p trained_models/kang_control/
python src/train_vega_kang_control.py
echo "DONE"
echo "Finished training all models."

