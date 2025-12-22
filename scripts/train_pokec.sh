# Train on 2x GPUs, test on CPU.
python nodecls/train.py \
  --data-path ./data/pokec/pokec.npz \
  --max-epochs 2000 \
  --train-batch-size 1632803 \
  --test-batch-size 1632803 \
  --num-classes 2 \
  --num-workers 8 \
  --model-type gnn \
  --eval-metric accuracy \
  --checkpoint-path ./work_dirs/pokec/gnn/ \
  --hidden-channels 256 \
  --num-layers 7 \
  --input-dropout 0.0 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --num-devices 2 \
  --test-after-train
