### Description
This is the core code of MCF-MOE
#### Dependecies
``` # requirements: 
    torch==2.7.0
    transformers==4.51.3
    dm-tree
    fastmoe   

```
#### Pretraining Transformer-XL on Enwik8 and Wikitext103: 

``` # pretrain: 
bash code/script/pretrain/pretrain_script.sh
```


``` # finetune: 
bash code/script/finetune/finetune_script.sh
```
