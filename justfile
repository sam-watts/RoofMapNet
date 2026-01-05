
recompile:
    uv pip compile requirements.in -o requirements.txt

sync:
    uv pip sync requirements.txt

# Prepare training data from RID2 edge labels
prepare-data:
    python -m roofmapnet.train.prepare_data

# Prepare data with visualization
prepare-data-viz:
    python -m roofmapnet.train.prepare_data --visualize

# Prepare data with custom paths
prepare-data-custom INPUT OUTPUT:
    python -m roofmapnet.train.prepare_data --input-dir {{INPUT}} --output-dir {{OUTPUT}}

# Train model for full run
train:
    @PYTHONWARNINGS='ignore::RuntimeWarning' python -m roofmapnet.train.main \
        --data-root rid2_edges/ \
        --image-dir ~/datasets/roof_information_dataset_2/images/

# Train model for 1 epoch (quick test)
train-test:
    @PYTHONWARNINGS='ignore::RuntimeWarning' python -m roofmapnet.train.main \
        --data-root rid2_edges/ \
        --image-dir ~/datasets/roof_information_dataset_2/images/ \
        --resume-from pretrained_models/roofmapnet.pth \
        --batch-size 1 \
        --max-epochs 1