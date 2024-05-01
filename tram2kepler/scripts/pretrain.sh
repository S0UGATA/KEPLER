TOTAL_UPDATES=125000                                    # Total number of training steps
WARMUP_UPDATES=10000                                    # Warmup the learning rate over this many updates
LR=6e-04                                                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=3                                         # Batch size.
NUM_NODES=1			                                        # Number of machines
ROBERTA_PATH="../data/keplerModels/KEPLERforNLP.pt"     # Path to the original roberta model
CHECKPOINT_PATH="../data/checkpoints"                   # Directory to store the checkpoints
UPDATE_FREQ="$((784 / NUM_NODES)) "                   # Increase the batch size

DATA_DIR="../data/output/single"

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
KE_DATA="$($DATA_DIR)/KE1_0:"

DIST_SIZE="$((NUM_NODES))"

python -m fairseq_cli.train $DATA_DIR/MLM-bin --KEdata $KE_DATA --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES \
        --tokens-per-sample 512 \
        --task MLMetKE \
        --sample-break-mode complete \
        --required-batch-size-multiple 1 \
        --arch roberta_base \
        --criterion MLMetKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq "$UPDATE_FREQ" \
        --negative-sample-size 1 --ke-model TransE \
        --init-token 0 \
        --separator-token 2 \
        --gamma 4 --nrelation 822 \
        --skip-invalid-size-inputs-valid-test \
        --fp16 --fp16-init-scale 2 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --reset-optimizer --distributed-world-size "${DIST_SIZE}" --ddp-backend no_c10d --distributed-port 23456 \
        --log-format simple --log-interval 1 \
        #--relation-desc  #Add this option to encode the relation descriptions as relation embeddings (KEPLER-Rel in the paper)