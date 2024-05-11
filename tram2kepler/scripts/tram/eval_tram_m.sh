export LD_LIBRARY_PATH=/home/sougata/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# mkdir scibert_multi_label_model
# wget https://ctidtram.blob.core.windows.net/tram-models/multi-label-20230803/config.json -O scibert_multi_label_model/config.json
# wget https://ctidtram.blob.core.windows.net/tram-models/multi-label-20230803/pytorch_model.bin -O scibert_multi_label_model/pytorch_model.bin

python eval_tram_m.py