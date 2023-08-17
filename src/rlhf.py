import gc
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from tap import Tap
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

REWARD_PROD_CONST = 4
REWARD_BASELINE = -20


class Arguments(Tap):
    model_name_or_path: str
    reward_model_name_or_path: str
    dataset_path: str
    num_epochs: int
    output_dir: str

    def configure(self) -> None:
        self.add_argument("--model_name_or_path", type=str, required=True)
        self.add_argument("--reward_model_name_or_path", type=str, required=True)
        self.add_argument("--dataset_path", type=str, required=True)
        self.add_argument("--num_epochs", type=int, required=True)
        self.add_argument("--output_dir", type=str, required=True)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(args: Arguments) -> None:
    device = torch.device("cuda:0")
    
    # define reward model
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name_or_path
    )
    reward_model.to(device)
    reward_model.eval()

    with open(os.path.join(args.model_name_or_path, "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # define llms
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    device_map = {"": 0}

    base_model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map=device_map, torch_dtype=torch.bfloat16
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, quantization_config=bnb_config
    )
    model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_config["base_model_name_or_path"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # define dataset
    dataset_df = pd.read_json(args.dataset_path, lines=True)
    dataset = Dataset.from_pandas(dataset_df)
    texts = [t.split("### Output\n")[0] + "### Output\n" for t in dataset["text"]]
    dataset = Dataset.from_dict({"query": texts})
    dataset = dataset.map(lambda sample: tokenizer(sample["query"], truncation=True))

    # define rlhf with ppo settings
    ppo_config = {
        "batch_size": 2,
        "learning_rate": 1e-6,
        "mini_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "optimize_cuda_cache": True,
        "log_with": "wandb",
    }

    config = PPOConfig(**ppo_config)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 50,
        "eos_token_id": -1,
    }

    ppo_trainer = PPOTrainer(
        config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    output_min_length = 30
    output_max_length = 156
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for _ in range(args.num_epochs):
        for ite, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = [
                torch.tensor(i).to(f"cuda:{model.current_device}")
                for i in batch["input_ids"]
            ]

            model.gradient_checkpointing_disable()
            model.pretrained_model.config.use_cache = True

            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)

            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            rm_inputs = rm_tokenizer(texts, padding=True, return_tensors="pt").to(
                device
            )

            with torch.inference_mode():
                rm_outputs = reward_model(**rm_inputs)

            rewards = [
                F.softmax(output, dim=0)[reward_model.config.label2id["quality"]]
                * REWARD_PROD_CONST
                - REWARD_BASELINE
                for output in rm_outputs.logits
            ]

            model.gradient_checkpointing_enable()
            model.pretrained_model.config.use_cache = False

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            if ite > 1 and (ite + 1) % 200 == 0:
                ppo_trainer.save_pretrained(
                    os.path.join(args.output_dir, f"checkpoint-{ite}")
                )

            gc.collect()
            torch.cuda.empty_cache()

        ppo_trainer.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
