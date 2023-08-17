import json
import os

from datasets import Dataset, load_dataset
from tap import Tap


def generate_prompt(instruction: str, inputs: str, outputs: str) -> str:
    return f"""### Instruction
{instruction}

### Input
{inputs}

### Output
{outputs}
"""


class Arguments(Tap):
    datasets_name: str
    train_size: float
    save_path: str

    def configure(self) -> None:
        self.add_argument("--datasets_name", type=str, required=True)
        self.add_argument("--train_size", type=float, required=True)
        self.add_argument("--save_path", type=str, required=True)


def main(args: Arguments) -> None:
    dataset = load_dataset(args.datasets_name, split="train")
    dataset_df = dataset.to_pandas()

    train_num = int(len(dataset_df) * args.train_size)
    with open(os.path.join(args.save_path, "train.jsonl"), "w") as f:
        for obj in Dataset.from_pandas(dataset_df.iloc[:train_num]):
            json.dump(
                {
                    "text": generate_prompt(
                        obj["instruction"], obj["input"], obj["output"]
                    )
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    with open(os.path.join(args.save_path, "eval.jsonl"), "w") as f:
        for obj in Dataset.from_pandas(dataset_df.iloc[train_num:]):
            json.dump(
                {
                    "text": generate_prompt(
                        obj["instruction"], obj["input"], obj["output"]
                    )
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)
