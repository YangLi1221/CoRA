# CoRA
Codes and datasets for our paper "Improving Long-Tail Relation Extraction with Collaborating Relation-Augmented Attention"

If you use the code, please cite the following [paper](https://arxiv.org/pdf/2010.03773.pdf):

```
@article{li2020improving,
  title={Improving Long-Tail Relation Extraction with Collaborating Relation-Augmented Attention},
  author={Li, Yang and Shen, Tao and Long, Guodong and Jiang, Jing and Zhou, Tianyi and Zhang, Chengqi},
  journal={arXiv preprint arXiv:2010.03773},
  year={2020}
}
```

## Requirements

The model is implemented using tensorflow. The versions of packages used are shown below.

* tensorflow = 1.4.1
* numpy = 1.13.3
* scipy = 0.19.1

## Initialization

First unzip the `./raw_data/data.zip` and put all the files under `./raw_data`. Once the original raw text corpus data is in `./raw_data`, run

    python script/initial_CoRA.py

## Train the model
For CoRA,

    PYTHONPATH=. python script/train.py

## Evaluate the model

Run various evaluation by specifying `--mode` in commandline, see the paper for detailed description for these evaluation methods.

    PYTHONPATH=. python scripts/evaluate.py --mode [test method: pr, pone, ptwo, pall] --test-single --test_start_ckpt [ckpt number to be tested] --model [cnn_hier or pcnn_hier]

The logits are saved at `./outputs/logits/`. To see the PR curve, run the following command which directly `show()` the curve, and you can adjust the codes in `./scripts/show_pr.py` for saving the image as pdf file or etc. :
    
    python scripts/show_pr.py [path/to/generated .npy logits file from evaluation]

## Pretrained models

The pretrained models is already saved at `./outputs/ckpt/`. To directly evaluate on them, run the following command:

    PYTHONPATH=. python scripts/evaluate.py --mode [test method: hit_k_100, hit_k_200, pr, pone, ptwo, pall] --test_single --test_start_ckpt 0 --model [cnn_hier or pcnn_hier]

And PR curves can be generated same way as above.

## The results of the released checkpoints

As this toolkit is reconstructed based on the original code and the checkpoints are retrained on this toolkit, the results of the released checkpoints are comparable with the reported ones.

### The Main Experiments

* AUC

|  Model   | AUC  |
|  ----  | ----  | 
| CoRA  | 0.53 |


### The Auxiliary Experiments

* Hits@N(<100)

|  micro   | 10  |  15 | 20 |
|  ----  | ----  | ---  | ---   |
| CoRA  | 61.1 | 66.6  |  83.3 |


|  macro   | 10  |  15 | 20 |
|  ----  | ----  | ---  | ---   |
| CoRA  | 66.7 | 72.2  |  87.0 |

* Hits@N(<200)

|  micro   | 10  |  15 | 20 |
|  ----  | ----  | ---  | ---   |
| CoRA  | 75.9 | 79.3 |  89.7 |


|  macro   | 10  |  15 | 20 |
|  ----  | ----  | ---  | ---   |
| CoRA  | 72.7 | 77.3  |  89.3 |


* pone

|  P@N   | 100  |  200 | 300| Mean|
|  ----  | ----  | ---  | ---   | ---  |
| CoRA  | 94.0 | 90.5 |  82.0 | 88.8 |

* ptwo

|  P@N   | 100  |  200 | 300| Mean|
|  ----  | ----  | ---  | ---   | ---  |
| CoRA  | 98.0 | 91.0 |  86.3 |  91.8 |

* pall

|  P@N   | 100  |  200 | 300| Mean|
|  ----  | ----  | ---  | ---   | ---  |
| CoRA  | 98.0 | 92.5  |  88.3 |  92.9 |


# Acknowledgement 
This repo is mainly based on [HNRE](https://github.com/thunlp/HNRE) from [@SillyXu](https://github.com/THUCSTHanxu13)
