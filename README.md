# Fine-tuning Large Language Models for Japanese


## Prepare datasets

```bash
$ python src/prepare_datasets.py --datasets_name kunishou/databricks-dolly-15k-ja --train_size 0.8 --save_path data/
```


## Supervised Fine-Tuning

```bash
$ python src/supervised_fine_tuning.py --dataset_path data/train.jsonl --base_model_name cyberagent/open-calm-7b --output_path outputs/
```


## Reinforcement Learning from Human Feedback (RLHF)
For simplicity, I use a trained reward model (`theblackcat102/reward-deberta-v3-large-aspect`).

```bash
$ python src/rlhf.py --model_name_or_path outputs/final_checkpoint/ --reward_model_name_or_path theblackcat102/reward-deberta-v3-large-aspect --dataset_path data/eval.jsonl --num_epochs 5 --output_dir outputs/rlhf
```
