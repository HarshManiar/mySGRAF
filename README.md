# SGRAF
*This is my attend at recreating and improving the PyTorch implementation for AAAI2021 paper of [**“Similarity Reasoning and Filtration for Image-Text Matching”**]* 
*Detailed instructions are provided to recreate the author's results as well as my results for this paper* 

## Introduction

**The framework of SGRAF:**

<img src="./fig/model.png" width = "100%" height="50%">

## Requirements 
We recommended the following dependencies:

*  Python 3.6  
*  [PyTorch (>=0.4.1)](http://pytorch.org/)    
*  [NumPy (>=1.12.1)](http://www.numpy.org/)   
*  [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)  
[Note]: The code applies to ***Python3.6 + Pytorch1.7***.


## Download data and vocab
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, which can be downloaded by using:

```bash
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```

## Pre-trained models and evaluation
Modify the **model_path**, **data_path**, **vocab_path** in the `evaluation.py` file. Then run `evaluation.py`:

```bash
python evaluation.py
```

Note that  `fold5=False` for flickr30K. Pretrained models and Log files can be downloaded from [Flickr30K_SGRAF](https://drive.google.com/file/d/1OBRIn1-Et49TDu8rk0wgP0wKXlYRk4Uj/view?usp=sharing) 

## Training new models from scratch
Modify the **data_path**, **vocab_path**, **model_name**, **logger_name** in the `opts.py` file. Then run `train.py`:

```bash
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --lr_update 30 --module_name SGR
(For SAF) python train.py --data_name f30k_precomp --num_epochs 30 --lr_update 20 --module_name SAF
```

## Training new models using our improvements
Modify the **data_path**, **vocab_path**, **model_name**, **logger_name** in the `opts.py` file. Then run `train.py`:

```bash
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --lr_update 25 --learning_rate 0.0003 --module_name SGR
(For SAF) python train.py --data_name f30k_precomp --num_epochs 20 --lr_update 15 --learning_rate 0.0003 --module_name SAF
```


## Reference

Reference to the original paper:

    @inproceedings{Diao2021SGRAF,
      title={Similarity Reasoning and Filtration for Image-Text Matching},
      author={Diao, Haiwen and Zhang, Ying and Ma, Lin and Lu, Huchuan},
      booktitle={AAAI},
      year={2021}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).  


