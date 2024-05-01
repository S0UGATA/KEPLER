export LD_LIBRARY_PATH=/home/sougata/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
