### Preprocessing
To extract features for training, including the markdown-only dataframes and sampling the code cells needed for each note book, simply run:

```$ python preprocess.py```

### Training method 1
```$ python -m torch.distributed.run --nnodes=1 --node_rank=0 --nproc_per_node=2 train_md_code_pair.py```


### Training method 2
```$ python -m torch.distributed.run --nnodes=1 --node_rank=0 --nproc_per_node=2 train_md_sampled_codes.py --model_name_or_path codebert-base --md_max_len 64 --total_max_len 512 --batch_size 80 --accumulation_steps 4 --epochs 15 --n_workers 8 ```

For evalutaing run train_md_code_pair.py or train_md_sampled_codes.py with flag --do_eval
