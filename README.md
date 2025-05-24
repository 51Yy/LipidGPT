# LipidGPT

## Acknowledgements
Thanks to the authors of "MolGPT: Molecular Generation Using a Transformer-Decoder Model" for sharing their code (https://doi.org/10.1021/acs.jcim.1c00600). Their work laid the foundation for LipidGPT, and we built this project based on their original codebase.

## Pre-training
python train/train.py --data_name <data_name> --augmentation 5 --max_len 140 --batch_size 256 --max_epochs 100 --frac 0.9 --path <path_to_data_files>

## Fine-tuning
python train/fine_tune.py --mpath <model_path>  --data_name <finetune_data> --augmentation 5 --max_len 140 --batch_size 32 --max_epochs 100 --custom_chars <custom_chars> --vocab_size <vocab_size> --path <path_to_data_files> --pre_data <path_to_predata_files> --frac 0.9

## Generation
python train/generate.py --mpath <model_path> --gen_size 5000 --vocab_size <vocab_size> --block_size 140 --ori_data <ori_data> --custom_chars <custom_chars> --max_len 140 --augmentation 5 --data_name <data_name> --path <path_to_data_files> --temp 0.9
