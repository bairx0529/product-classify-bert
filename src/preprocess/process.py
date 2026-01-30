from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

from configuration import config


def process_data():
    print("开始处理数据")

    # 1) 读取文件
    dataset_dict = load_dataset(
        "csv",
        data_files={
            "train": str(config.RAW_DATA_DIR / "train.txt"),
            "test": str(config.RAW_DATA_DIR / "test.txt"),
            "valid": str(config.RAW_DATA_DIR / "valid.txt"),
        },
        delimiter="\t",
    )

    # 2) 清洗 + 过滤（避免 None / 空字符串 / 多余空白）
    def clean_example(ex):
        # 防御：某些行可能缺字段或为 None
        lbl = "" if ex.get("label") is None else str(ex["label"]).strip()
        txt = "" if ex.get("text_a") is None else str(ex["text_a"]).strip()
        ex["label"] = lbl
        ex["text_a"] = txt
        return ex

    dataset_dict = dataset_dict.map(clean_example)
    dataset_dict = dataset_dict.filter(lambda x: x["label"] != "" and x["text_a"] != "")

    # 3) 分词（先保留 label 列，移除 text_a）
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / "bert-base-chinese")

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text_a"],
            truncation=True,
            padding="max_length",
            max_length=config.SEQ_LEN,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    dataset_dict = dataset_dict.map(tokenize, batched=True, remove_columns=["text_a"])

    # 4) 处理 label：字符串 -> int -> ClassLabel
    #    注意：cast_column 只能对 int label 生效，不能直接把字符串 cast 成 ClassLabel
    all_labels = dataset_dict["train"].unique("label")  # 这里还是字符串列表
    all_labels = sorted([str(x).strip() for x in all_labels if str(x).strip() != ""])
    print("类别总数：", len(all_labels))

    label2id = {name: i for i, name in enumerate(all_labels)}
    class_label = ClassLabel(names=all_labels)

    def map_label(ex):
        lbl = str(ex["label"]).strip()
        if lbl not in label2id:
            raise ValueError(f"Unknown label: {lbl}")
        ex["label"] = label2id[lbl]  # 转成 int
        return ex

    dataset_dict = dataset_dict.map(map_label)
    dataset_dict = dataset_dict.cast_column("label", class_label)

    # 5) 保存数据集
    dataset_dict["train"].save_to_disk(config.PROCESSED_DATA_DIR / "train")
    dataset_dict["test"].save_to_disk(config.PROCESSED_DATA_DIR / "test")
    dataset_dict["valid"].save_to_disk(config.PROCESSED_DATA_DIR / "valid")

    print("数据处理完成")
