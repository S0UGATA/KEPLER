export LD_LIBRARY_PATH=/home/sougata/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python -m fairseq_cli.eval_lm /home/sougata/projects/MyKEPLER/tram2kepler/data/output/single/MLM-bin \
    --path /home/sougata/projects/MyKEPLER/tram2kepler/data/checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --output-word-probs --output-word-stats \
    --log-format='simple' --bpe='gpt2' \
