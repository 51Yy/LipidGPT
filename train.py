import pandas as pd
import numpy as np
import argparse
import os
import re
import json
from random import shuffle

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
from utils import set_seed
from do_augmentation import augment_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SMILES Generator Training Script')


    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')

    parser.add_argument('--data_name', type=str, default='allILs',
                        help="name of the dataset to train on", required=False)

    ### transformer parameters ###
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="dimension of embedding vectors", required=False)
    ### transformer parameters ###


    ### training parameters ###
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="maximum number of training epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=256,  
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,  
                        default=6e-4, help="learning rate", required=False)
    ### training parameters ###
    

    ### 添加的参数 ###
    parser.add_argument('--path', type=str,
                        default=None, help="path to the data named as date_name")
    parser.add_argument('--custom_chars', type=str,
                        default=None, help="use custom chars")
    parser.add_argument('--min_len', type=int,
                        default=1, help="min length of smiles")
    parser.add_argument('--max_len', type=int,
                        default=140, help="max length of smiles")
    parser.add_argument('--augmentation', type=int,
                        default=10, help="value to augment the dataset")
    parser.add_argument('--verbose', type=str,
                        default='True', help="verbose")
    parser.add_argument('--frac', type=float,
                        default=0.9, help="fraction of training data to use")
    ### 添加的参数 ###
    
    args = parser.parse_args()

    print('Arguments list:')
    for k, v in vars(args).items():
        print(k, ' ', v)
    print()
       
    set_seed(42) 

    # define the path to the data files
    dir_path = f'./results{args.path}'
    save_name = f'{args.min_len}_{args.max_len}_x{args.augmentation}'
    data_dir = f'{dir_path}/data/{args.data_name}/{save_name}/'
    model_dir = f'{dir_path}/models/{args.data_name}/{save_name}/'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    ### 1. load data ###
    print("==========Loading data==========")

    data = pd.read_csv(f'./datasets{args.path}/{args.data_name}.csv')
    data.columns = data.columns.str.lower()

    # drop long smiles, keep smiles with length <= 140
    if args.max_len:
        print('Filtering SMILES by length...')
        data = data[data['smiles'].str.len() <= args.max_len]
        data = data.reset_index(drop=True)
        print('Number of SMILES after length filtering: ', len(data))

    # Split data into train and validation
    train_data = data.sample(frac=args.frac, random_state=42)
    val_data = data.drop(train_data.index)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    if not os.path.isfile(f'{data_dir}{args.data_name}_tr.csv'):
        train_data.to_csv(f'{data_dir}{args.data_name}_tr.csv', index=False)
        val_data.to_csv(f'{data_dir}{args.data_name}_val.csv', index=False)

    print('Number of ori training examples: ', len(train_data))
    print('Number of ori validation examples: ', len(val_data))

    # 读取SMILES
    tr_smiles = train_data['smiles'].tolist()
    val_smiles = val_data['smiles'].tolist()
    
    print('==========Finished loading data==========\n')

    ### 2. augment data before padding ###
    """
        Augment separately the training and validation splits.
        It's important to do those steps separetely in order
        to avoid to have the same molecule represented in both splits
    """
    if args.augmentation > 0 :

        print(f'==========Augmenting data {args.augmentation}-fold==========')

        tr_aug = augment_dataset(
            tr_smiles, augmentation=args.augmentation, min_len=args.min_len, max_len=args.max_len, verbose=args.verbose)
        val_aug = augment_dataset(
            val_smiles, augmentation=args.augmentation, min_len=args.min_len, max_len=args.max_len, verbose=args.verbose)

        # Merge with the original data and shuffle
        tr_smiles = list(set(tr_smiles + tr_aug))
        shuffle(tr_smiles)
        val_smiles = list(set(val_smiles + val_aug))
        shuffle(val_smiles)

        # Save augmented data
        train_data = pd.DataFrame({'smiles': tr_smiles})
        val_data = pd.DataFrame({'smiles': val_smiles})
        train_data.to_csv(
            f'{data_dir}{args.data_name}_aug{args.augmentation}_tr.csv', index=False)
        val_data.to_csv(
            f'{data_dir}{args.data_name}_aug{args.augmentation}_val.csv', index=False)

        if args.verbose:
            print(f'Augmented training set size: {len(tr_smiles)}')
            print(f'Augmented validation set size: {len(val_smiles)}')

    print('=====Finished augmenting data=====\n')

    ### 3. pre-processing data ###
    print('==========Pre-Processing data===========')

    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)  # regex for tokenizing smiles

    if args.max_len:
        max_len = args.max_len
        print(f"Using fixed max SMILES length: {max_len}")
    else:
        lens = [len(regex.findall(i.strip()))
                for i in tr_smiles + val_smiles]  # i.strip()去掉首尾空格
        max_len = max(lens)
        print(f'Detected max SMILES length: {max_len}')

    whole_string = ' '.join(tr_smiles + val_smiles)
    whole_string = sorted(list(set(regex.findall(whole_string))))  # list
    whole_string = ['<'] + whole_string # 添加padding token, 且padding token的index为0

    # build vocab
    stoi = {ch: i for i, ch in enumerate(whole_string)}  # dict
    itos = {i: ch for i, ch in enumerate(whole_string)}  # dict

    if args.custom_chars is not None:
        print("Using custom chars")
        whole_string = json.load(
            open(f'./datasets{args.path}/char/{args.custom_chars}_aug{args.augmentation}_chars.json', 'r'))
        stoi = {ch: i for i, ch in enumerate(whole_string)}
        itos = {i: ch for i, ch in enumerate(whole_string)}
    else:
        with open(f'./datasets{args.path}/char/{args.data_name}_aug{args.augmentation}_chars.json', 'w') as f:
            json.dump(whole_string, f)
        
        with open(f'./datasets{args.path}/char/{args.data_name}_aug{args.augmentation}_stoi.json', 'w') as f:
            json.dump(stoi, f)
        
        with open(f'./datasets{args.path}/char/{args.data_name}_aug{args.augmentation}_itos.json', 'w') as f:
            json.dump(itos, f)

    print('Chars in vocab: ', itos)
    print('Padding token: ', itos[0], stoi['<'])
    print('=====Finished pre-processing data=====\n')

    ### 4. training ###
    train_dataset = SmileDataset(args, tr_smiles, whole_string, max_len)
    valid_dataset = SmileDataset(args, val_smiles, whole_string, max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(mconf)

    print('Starting training...')

    train_config = TrainerConfig(
        max_epochs=args.max_epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        lr_decay=True, 
        warmup_tokens=0.1*len(train_data)*max_len, 
        final_tokens=args.max_epochs*len(train_data)*max_len,
        num_workers=0, 
        ckpt_path=f'{model_dir}{args.data_name}_{args.augmentation}.pt')  # block_size=train_dataset.max_len, generate=False 在训练的时候用不上

    trainer = Trainer(model, train_dataset, valid_dataset, train_config)
    train_losses, val_losses = trainer.train()

    np.savetxt(
        f'{model_dir}{args.data_name}_{args.augmentation}_tr_losses.txt', train_losses)
    np.savetxt(
        f'{model_dir}{args.data_name}_{args.augmentation}_val_losses.txt', val_losses)

    print('=====Finished training=====')
