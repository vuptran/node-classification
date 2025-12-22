# Train on 1x GPU, test on GPU.
python nodecls/train.py \
  --data-path ./data/ogbn/arxiv/ogbn-arxiv.npz \
  --max-epochs 2000 \
  --train-batch-size 169343 \
  --test-batch-size 169343 \
  --num-classes 40 \
  --num-workers 8 \
  --model-type gnn \
  --eval-metric accuracy \
  --checkpoint-path ./work_dirs/ogbn_arxiv/gnn/ \
  --hidden-channels 256 \
  --num-layers 4 \
  --input-dropout 0.1 \
  --dropout 0.5 \
  --learning-rate 0.0005 \
  --num-devices 1 \
  --test-after-train \
  --test-on-gpu
