CUDA_VISIBLE_DEVICES=0,1,2 mmf_run dataset=sal_textvqa \
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



