#!/usr/bin/env bash

echo "Creating subdirectories"
mkdir -p ./paper/figures/
mkdir -p ./trained_models/
echo "Training model for Kang PBMC (Fig1)..."
python src/kang_pbmc.py
echo "DONE"
echo "Training model for SuppFig1..."
mkdir -p ./trained_models/compare_models/
python src/train_vega_suppFig1.py
python src/train_sparse_vae_suppFig1.py
python src/train_deep_decoder_suppFig1.py
python src/train_vanilla_vae_suppFig1.py
echo "DONE"
echo "Training model for MIX-Seq (Fig2)..."
mkdir -p ./trained_models/mixseq/
python src/mixseq_vega.py
echo "DONE"
echo "Training model for GBM (Fig2)..."
python src/gbm_vega.py
echo "DONE"
echo "Training model for organoid (Fig3)..."
python src/organoid_vega.py
echo "DONE"
echo "Training model for Fig4."
mkdir -p ./trained_models/vega_ct_out/
python src/kang_ct_out.py
python src/kang_ctrl_vega.py
echo "DONE"
echo "Training model for SuppFig2..."
mkdir -p ./trained_models/retina/
python src/retina_batch_correction.py
echo "DONE"
echo "Training model for SuppFig3..."
python src/kang_dec_binary.py
echo "DONE"
echo "Training model for SuppFig8..."
python src/train_kang_pbmc_count.py
echo "DONE"
echo "Training model for SuppFig9..."
python src/intercode_vega_comp.py
echo "DONE"
echo "Training model for SuppFig10..."
mkdir -p ./trained_models/kang_l1/
python src/kang_l1.py
echo "DONE"
echo "Training model for SuppFig11..."
mkdir -p ./trained_models/node_experiment/
python src/search_extra_nodes.py -n 1
python src/search_extra_nodes.py -n 4
python src/search_extra_nodes.py -n 8
python src/search_extra_nodes.py -n 16
python src/search_extra_nodes.py -n 32
python src/search_extra_nodes.py -n 64
echo "DONE"
echo "Training model for SuppFig12..."
mkdir -p ./trained_models/corr_z/
python src/kang_corr_z.py
echo "DONE"
echo "Finished training all models."

