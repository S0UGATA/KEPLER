export LD_LIBRARY_PATH=/home/sougata/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python -m transformers.convert_roberta_original_pytorch_checkpoint_to_pytorch \
			--roberta_checkpoint_path /home/sougata/projects/MyKEPLER/tram2kepler/data/checkpoints/convert/single/input \
			--pytorch_dump_folder_path /home/sougata/projects/MyKEPLER/tram2kepler/data/checkpoints/convert/single/output/

python -m transformers.convert_roberta_original_pytorch_checkpoint_to_pytorch \
			--roberta_checkpoint_path /home/sougata/projects/MyKEPLER/tram2kepler/data/checkpoints/convert/multi/input \
			--pytorch_dump_folder_path /home/sougata/projects/MyKEPLER/tram2kepler/data/checkpoints/convert/multi/output/