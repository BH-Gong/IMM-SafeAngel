# Work

## Version (Conformed version combination)
Ubuntu 24.04, 22.04\
Python 3.12

## Git clone (Ubuntu)
```
git clone https://github.com/BH-Gong/AI-Champion.git
```

## Library
```
pip install -r requirements.txt
```

## Config
```
 {
    "mode": "train",                                    -> Choose the run type. ("train" or "inference")
    "model_type": "BEATs",                              -> Choose the model frame. ("BEATs" or "CNN" of "Transformer")

    "model_pt_path": "./pt/BEATs_iter3_plus_AS2M.pt",   -> Insert the Pre-trained model of BEATs (Find details under bellow.)
    "classifier_pt_path":"./result/best.pt",            -> Select the trained model directory of pt to inference.

    "train_file_list": "./data/flist/train",            -> Train data path flist directory.
    "valid_file_list": "./data/flist/valid",            -> Valid data path flist directory.
    "test_file_list": "./data/flist/test",              -> Test data path flist directory.

    "epochs": 300,                                      -> Number of epochs.
    "save_term": 10,                                    -> Number of save term.
    "lr": 5e-6,                                         -> Initial learning rate.
    "min_lr": 1e-10,                                    -> Minimum learning rate. (for scheduler)
    "optimizer": "AdamW",                               -> Optimizer.
    "criterion": "CrossEntropyLoss",                    -> Criterion.
    "scheduler": "CosineAnnealingLR",                   -> Scheduler.
    "T_max": 300                                        -> Maximum epoch for scheduler calulation.
}

```

## Directory setup
```
AICH
  └data
    └flist
      └train
        └class_1.flist
        ... (amount of classes)
        └class_n.flist
      └test
        └class_1.flist
        ... (amount of classes)
        └class_n.flist
      └valid
        └class_1.flist
        ... (amount of classes)
        └class_n.flist
  └pt
    └pre_trained_model_weight.pt
  └result
```

## Run with single gpu
```
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 run.py
```

you can change the nproc_per_node number according to your training hardware. (number of gpus)



# BEATs

[**BEATs**](https://arxiv.org/abs/2212.09058): **Audio Pre-Training with Acoustic Tokenizers**

Official PyTorch implementation and pretrained models of BEATs

## Pre-Trained and Fine-Tuned Tokenizers and Models
Iterations  | Tokenizer  | Pre-Trained Model | AudioSet Fine-Tuned Model 1 | AudioSet Fine-Tuned Model 2
|---|---|---|---|---
Iter1  |  Random Projection | [BEATs_iter1](https://1drv.ms/u/s!AqeByhGUtINrgcpmY7IHhgc9q0pT7Q?e=uQuisJ)  | [Fine-tuned BEATs_iter1 (cpt1)](https://1drv.ms/u/s!AqeByhGUtINrgcpuRfRZmco2XulmFw?e=f2INHa) | [Fine-tuned BEATs_iter1 (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpyMlTmnRh0Wp_Qgg?e=sgzv8H) | 
Iter2  |  [Tokenizer_iter2](https://1drv.ms/u/s!AqeByhGUtINrgcpnFGsfd_buKng5Pw?e=avWBJw)| [BEATs_iter2](https://1drv.ms/u/s!AqeByhGUtINrgcpwwEGgUyiI-jQyQw?e=1rP1RI)  | [Fine-tuned BEATs_iter2 (cpt1)](https://1drv.ms/u/s!AqeByhGUtINrgcp4l547zKa7xPqy8w?e=rsLdPr) | [Fine-tuned BEATs_iter2 (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcp5APbt_2bdIQvX0w?e=2cd2ry) | 
Iter3  |  [Tokenizer_iter3](https://1drv.ms/u/s!AqeByhGUtINrgcp1DEzUBtzHapxcqw?e=JZI5Uf)| [BEATs_iter3](https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?e=qezPJ5)  | [Fine-tuned BEATs_iter3 (cpt1)](https://1drv.ms/u/s!AqeByhGUtINrgcplb48ll1zIt82eWQ?e=XyxrX7) | [Fine-tuned BEATs_iter3 (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcptb4S-CeJnlJGtZA?e=2FyDy3) | 
Iter3+  |  [Tokenizer_iter3+ (AS20K)](https://1drv.ms/u/s!AqeByhGUtINrgcpz_SnXxs0SrwHEwA?e=14nugm)| [BEATs_iter3+ (AS20K)](https://1drv.ms/u/s!AqeByhGUtINrgcpvdNz8-aYim60CIg?e=53V8pg)  | [Fine-tuned BEATs_iter3+ (AS20K) (cpt1)](https://1drv.ms/u/s!AqeByhGUtINrgcp2YHUCT1uZx2Kysw?e=nvu1Dw) | [Fine-tuned BEATs_iter3+ (AS20K) (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcp092af0h7P3kXKFA?e=kUkPhN) | 
Iter3+  |  [Tokenizer_iter3+ (AS2M)](https://1drv.ms/u/s!AqeByhGUtINrgcppJUDx2TmXiIMFyQ?e=pJsOLl)| [BEATs_iter3+ (AS2M)](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf)  | [Fine-tuned BEATs_iter3+ (AS2M) (cpt1)](https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC) | [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea) | 

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) and [VQGAN](https://github.com/CompVis/taming-transformers) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


### Reference
If you find our work is useful in your research, please cite the following paper:
``` latex
@article{Chen2022beats,
  title = {BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author  = {Sanyuan Chen and Yu Wu and Chengyi Wang and Shujie Liu and Daniel Tompkins and Zhuo Chen and Furu Wei},
  eprint={2212.09058},
  archivePrefix={arXiv},
  year={2022}
}
```


