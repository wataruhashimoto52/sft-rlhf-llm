import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from tap import Tap
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


class Arguments(Tap):
    dataset_path: str
    base_model_name: str
    output_path: str

    def configure(self) -> None:
        self.add_argument("--dataset_path", type=str, required=True)
        self.add_argument("--base_model_name", type=str, required=True)
        self.add_argument("--output_path", type=str, required=True)


def main(args: Arguments) -> None:
    dataset = Dataset.from_pandas(pd.read_json(args.dataset_path, lines=True))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500,
    )

    max_seq_length = 512

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.model.save_pretrained(os.path.join(args.output_path, "final_checkpoint"))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
