# Train on 4x GPUs, test on CPU.
python nodecls/train.py --data-path ./data/ogbn/products/ogbn-products.npz --max-epochs 2000 --train-batch-size 2449029 --test-batch-size 2449029 --num-classes 47 --num-workers 8 --model-type gnn --eval-metric accuracy --checkpoint-path ./work_dirs/ogbn_products/gnn/ --hidden-channels 256 --num-layers 7 --input-dropout 0.1 --dropout 0.5 --learning-rate 0.001 --num-devices 4 --test-after-train

