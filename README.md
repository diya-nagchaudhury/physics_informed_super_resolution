# Training
python scripts/train.py --model residual --epochs 100 --batch_size 32

# Evaluation 
python scripts/evaluate.py --model residual --checkpoint best_model.pth --generate_report

# Visualization with energy maps
python scripts/visualize.py --mode energy_maps --checkpoint best_model.pth --samples 10