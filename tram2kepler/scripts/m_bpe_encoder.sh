python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json ../data/gpt2_bpe/encoder.json \
    --vocab-bpe ../data/gpt2_bpe/vocab.bpe \
    --inputs ../data/output/single/nodes.txt \
    --outputs ../data/output/single/nodes.bpe \
    --keep-empty \
    --workers 60