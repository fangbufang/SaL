## Separate and Locate: Rethink the Text in Text-based Visual Question Answering

1. Implementation of [Separate and Locate: Rethink the Text in Text-based Visual Question Answering](https://arxiv.org/abs/2308.16383)
2. Our code is based on  [MMF (or Pythia)](https://github.com/facebookresearch/mmf "MMF's Github repo").

### Requirements
1. Follow the installation steps in [MMF](https://github.com/facebookresearch/mmf "MMF's Github repo").
```
$ git clone https://github.com/fangbufang/SaL.git
$ cd SaL
$ pip install --editable .
```
2. See [`requirements.txt`](requirements.txt) for the required python packages and run to install them
3. Replace all */data/fcy/sal* in this project with the absolute path of the project on your machine
4. The t5-base model is used by default, if you want to use the t5-large model, please replace all *t5-base* with *t5-large*

### Data Setup
|  dataset   | data link | path|
|  ----  | ----  |---|
| textvqa  | [data](https://1drv.ms/f/s!AqrlHS5FqqinkC9d_yxl5ArrVMBI?e=NMVosY) |./data/textvqa|
| stvqa  | [data](https://1drv.ms/f/s!AqrlHS5FqqinkCnB3Y7T1IFalotx?e=NbPDlX) |./data/stvqa|
### TextVQA Training

```
#For textvqa t5-base
$ CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=sal_textvqa \
  model=sal \
  config=projects/sal/configs/textvqa/adam.yaml \
  env.save_dir=./save/sal_train\
  run_type=train_val \
  training.seed=1\
  training.num_workers=2\
  training.batch_size=36\
  training.max_updates=24000\
  training.evaluation_interval=1000\
  training.checkpoint_interval=1000\
  checkpoint.max_to_keep=5\
  optimizer.params.lr=2e-4\
  model_config.sal.mmt.dropout_rate=0.1\

#For textvqa with stvqa t5-base
$ CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=sal_textvqa \
  model=sal \
  config=projects/sal/configs/textvqa/with_stvqa.yaml \
  env.save_dir=./save/sal_train_withstvqa\
  run_type=train_val \
  training.seed=1\
  training.num_workers=2\
  training.batch_size=36\
  training.max_updates=24000\
  training.evaluation_interval=1000\
  training.checkpoint_interval=1000\
  checkpoint.max_to_keep=5\
  optimizer.params.lr=2e-4\
  model_config.sal.mmt.dropout_rate=0.1\

```

### ST-VQA Training

```
#For stvqa t5-base
$ CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=stvqa \
  model=sal \
  config=projects/sal/configs/stvqa/defaults.yaml \
  env.save_dir=./save/stvqa_sal_train\
  run_type=train_val \
  training.seed=1\
  training.num_workers=2\
  training.batch_size=36\
  training.max_updates=24000\
  training.evaluation_interval=1000\
  training.checkpoint_interval=1000\
  checkpoint.max_to_keep=5\
  optimizer.params.lr=2e-4\
  model_config.sal.mmt.dropout_rate=0.1\

#For stvqa with textvqa t5-base
$ CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=stvqa \
  model=sal \
  config=projects/sal/configs/stvqa/with_textvqa.yaml \
  env.save_dir=./save/stvqa_sal_train_withtextvqa\
  run_type=train_val \
  training.seed=1\
  training.num_workers=2\
  training.batch_size=36\
  training.max_updates=24000\
  training.evaluation_interval=1000\
  training.checkpoint_interval=1000\
  checkpoint.max_to_keep=5\
  optimizer.params.lr=2e-4\
  model_config.sal.mmt.dropout_rate=0.1\

```


### Evaluation

```
#For textvqa t5-base
$ CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=sal_textvqa \
  model=sal \
  config=projects/sal/configs/textvqa/adam.yaml \
  env.save_dir=./save/sal_train\
  run_type=val \
  training.seed=1\
  training.num_workers=2\
  training.batch_size=36\
  training.max_updates=24000\
  training.evaluation_interval=1000\
  training.checkpoint_interval=1000\
  checkpoint.max_to_keep=5\
  optimizer.params.lr=2e-4\
  model_config.sal.mmt.dropout_rate=0.1\
  checkpoint.resume_file=./save/textvqa/base/best.ckpt


```


### Inference

```
#For textvqa t5-base
$ CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=sal_textvqa \
  model=sal \
  config=projects/sal/configs/textvqa/adam.yaml \
  env.save_dir=./save/sal_train\
  run_type=test \
  training.seed=1\
  training.num_workers=2\
  training.batch_size=36\
  training.max_updates=24000\
  training.evaluation_interval=1000\
  training.checkpoint_interval=1000\
  checkpoint.max_to_keep=5\
  evaluation.predict=True\
  optimizer.params.lr=2e-4\
  model_config.sal.mmt.dropout_rate=0.1\
  checkpoint.resume_file=./save/textvqa/base/best.ckpt


```



###  Result 
move [ckpt files](https://1drv.ms/f/s!AqrlHS5FqqinkBot-EiQMBtefVMq?e=wtOUgQ) to ./save
|  dataset   | SaL-base  | SaL-large|
|  ----  | ----  |---|
| textvqa  | 62.42 (62.85) |63.88 (64.58)|
| stvqa  | 59.74(62.29) |61.45(64.16)|

### Bibtex
```
@inproceedings{10.1145/3581783.3611753,
author = {Fang, Chengyang and Li, Jiangnan and Li, Liang and Ma, Can and Hu, Dayong},
title = {Separate and Locate: Rethink the Text in Text-based Visual Question Answering},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3611753},
doi = {10.1145/3581783.3611753},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {4378–4388},
numpages = {11},
keywords = {textvqa, scene understanding, multimodal information},
}
```