# TacoLM
***T***empor***A***l ***CO***mmon Sense ***L***anguage ***M***odel

A variation of BERT that is aware of temporal common sense.

## Introduction
This is the code repository for our ACL 2020 paper [Temporal Common Sense Acquisition with Minimal Supervision](https://arxiv.org/pdf/2005.04304.pdf).
This package is built upon [huggingface/transformers](https://github.com/huggingface/transformers) at its April 2019 version. 

## Installation
- `pip install -r requirements.txt`
- `pip install --editable .`

## Out of the box
Here are some things you can do with this package out of the box.

### Train the main model 
- Access and download `data/tmp_seq_data` at [Google Drive](https://drive.google.com/drive/folders/1kx5Vc8iFYorWHrxHndkUzOssdeOm8oYC?usp=sharing) (4.6 G)
- run `sh train_taco_lm.sh`

The script is set to default parameters and will export the model to `models/`. You can configure differently by editing the script.

The training process will generate one directory to store the loss logs, as well as `NUM_EPOCH` directories for each epoch's model.
You will need to add BERT's `vocab.txt` to the epoch directories for evaluation. See more detail in the next section on pre-trained models.

The training data is pre-generated and formatted. More details [here](pattern_extraction.md).

### Experiments
You can download pre-trained models in `models/` at [Google Drive](https://drive.google.com/drive/folders/1kx5Vc8iFYorWHrxHndkUzOssdeOm8oYC?usp=sharing) (0.4 G each), 
or follow the training procedure in the previous section. 

#### General Usage
You can do many things with the model by just treating it as a set of transformer weights that fit exactly into a BERT-base model. Have an on-going project with BERT? Give it a try!

#### Intrinsic Experiments
The intrinsic evaluation relies on pre-formatted data.
- run `sh eval_intrinsic.sh`
- see `eval_results/intrinsic.txt` for results

#### TimeBank Experiment
- by default this requires the epoch 2 model. 
- run `sh eval_timebank.sh` to produce evaluation results on 3 different seeds. They are by default stored under `eval_results`
- run `python scripts/eval_timebank.py` to see result interpretations.

#### HiEVE Experiment
- by default this requires the epoch 2 model. 
- run `sh eval_hieve.sh` to produce eval results under `eval_results`
- run `python scripts/eval_hieve.py` to see interpretations.

#### MC-TACO Experiment
See [MC-TACO](https://github.com/CogComp/MCTACO). 
- use the augmented data under `data/mctaco-tcs`
- use the transformer weights of `taco_lm_epoch_2`

## Citation
See the following paper: 
```
@inproceedings{ZNKR20,
    author = {Ben Zhou, Qiang Ning, Daniel Khashabi and Dan Roth},
    title = {Temporal Common Sense Acquisition with Minimal Supervision},
    booktitle = {ACL},
    year = {2020},
}
```
