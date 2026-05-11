### `Dataset`

| 数据格式           | 类型参数 | 加载的指令                                              |
| ------------------ | -------- | ------------------------------------------------------- |
| CSV & TSV          | `csv`    | `load_dataset("csv", data_files="my_file.csv")`         |
| Text files         | `text`   | `load_dataset("text", data_files="my_file.txt")`        |
| JSON & JSON Lines  | `json`   | `load_dataset("json", data_files="my_file.jsonl")`      |
| Pickled DataFrames | `pandas` | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

当我们使用 `load_dataset()` 函数来加载 JSON 文件时，我们需要知道我们是在处理普通的 JSON（类似于嵌套字典）还是 JSON Lines（每一行都是一个 JSON）。

``` python
from datasets import load_dataset
# SQuAD-it 使用的是嵌套字典，所有文本都存储在 data 字段中。这意味着我们可以通过使用参数 field 来加载数据集
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# load_dataset() 函数的 data_files 参数非常灵活：可以是单个文件路径、文件路径列表或者是标签映射到文件路径的字典。你还可以根据 Unix shell 的规则，对符合指定模式的文件进行批量匹配（例如，你可以通过设置 data_files="*.JSON" 匹配目录中所有的 JSON 文件）。
# Datasets 实际上支持自动解压输入文件，所以我们可以跳过使用 gzip ，直接将 data_files 参数设置为压缩文件
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# 加载远程数据集
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

```
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip drugsCom_raw.zip
```

``` python
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t 在python中是制表符的意思
#  TSV 仅仅是 CSV 的一个变体，它使用制表符而不是逗号作为分隔符
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# 细看前几个例子
drug_sample[:3]
"""
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}
"""

# 验证 Unnamed: 0 列存储的是患者 ID 的猜想
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
    
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
 # 删除数据集中的所有condition 为 None的记录
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset = drug_dataset.map(lowercase_condition)

drug_dataset = drug_dataset.map(lambda x: {"review_length": len(x['review'].split())})

drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
# 处理评论中的 HTML 字符。我们可以使用 Python 的 html 模块来解码这些字符
import html
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

# 使用 Dataset.map() 函数时设定 batched=True 。该函数需要接收一个包含数据集字段的字典，字典的值是一个列表。
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)


def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )  # 例如，长度为250的一行数据，变成两行数据，直接拼接的话，长度会出现不一致
    # 提取新旧索引之间的映射
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
result = tokenize_and_split(drug_dataset["train"][:3])
"""
{'input_ids': [[101, 107, 1135, 1144, 1185, 1334, 2629, 117, 146, 102], [101, 1321, 1122, 1107, 4612, 1104, 1650, 12223, 8031, 102], [101, 126, 150, 1403, 1105, 9425, 9105, 107, 102], [101, 107, 1422, 1488, 1110, 9079, 1194, 1117, 2223, 102], [101, 1989, 1104, 1130, 19972, 11083, 119, 1284, 1245, 102], [101, 4264, 1165, 1119, 1310, 1142, 1314, 1989, 117, 102], [101, 1165, 1119, 1408, 1781, 1103, 2439, 13753, 1119, 102], [101, 1209, 1129, 1113, 119, 1370, 1160, 1552, 117, 102], [101, 1119, 1180, 6374, 1243, 1149, 1104, 1908, 117, 102], [101, 1108, 1304, 172, 14687, 1183, 117, 1105, 7362, 102], [101, 1111, 2212, 129, 2005, 1113, 170, 2797, 1313, 102], [101, 1121, 1278, 12020, 113, 1304, 5283, 1111, 1140, 102], [101, 119, 114, 146, 1270, 1117, 3995, 1113, 6356, 102], [101, 2106, 1105, 1131, 1163, 1106, 6166, 1122, 1149, 102], [101, 170, 1374, 1552, 119, 3969, 1293, 1119, 1225, 102], [101, 1120, 1278, 117, 1105, 1114, 2033, 1146, 1107, 102], [101, 1103, 2106, 119, 1109, 1314, 1160, 1552, 1138, 102], [101, 1151, 2463, 1714, 119, 1124, 1110, 150, 21986, 102], [101, 3048, 1167, 5340, 1895, 1190, 1518, 119, 1124, 102], [101, 1110, 1750, 6438, 113, 170, 1363, 1645, 114, 102], [101, 117, 1750, 172, 14687, 1183, 119, 1124, 1110, 102], [101, 11566, 1155, 1103, 1614, 1119, 1431, 119, 8007, 102], [101, 1117, 4658, 1110, 1618, 119, 1284, 1138, 1793, 102], [101, 1242, 1472, 23897, 1105, 1177, 1677, 1142, 1110, 102], [101, 1103, 1211, 3903, 119, 107, 102], [101, 107, 146, 1215, 1106, 1321, 1330, 9619, 14255, 102], [101, 4487, 17046, 117, 1134, 1125, 1626, 21822, 5120, 102], [101, 117, 1105, 1108, 1304, 2816, 118, 1304, 1609, 102], [101, 6461, 117, 12477, 1775, 126, 1552, 117, 1185, 102], [101, 1168, 1334, 3154, 119, 1252, 1122, 4049, 21055, 102], [101, 176, 2556, 13040, 1673, 117, 1134, 1110, 1136, 102], [101, 1907, 1107, 1646, 117, 1177, 146, 6759, 1106, 102], [101, 149, 1183, 9730, 1233, 117, 1272, 1103, 13288, 102], [101, 1132, 1861, 119, 1332, 1139, 1168, 17029, 2207, 102], [101, 117, 146, 1408, 149, 1183, 9730, 1233, 2411, 102], [101, 117, 1113, 1139, 1148, 1285, 1104, 1669, 117, 102], [101, 1112, 1103, 7953, 1163, 119, 1262, 1103, 1669, 102], [101, 5695, 1111, 1160, 2277, 119, 1332, 1781, 1103, 102], [101, 1248, 5246, 118, 1269, 1160, 2277, 119, 1262, 102], [101, 1208, 117, 1114, 1503, 5246, 1614, 1400, 1256, 102], [101, 4146, 118, 1139, 1503, 1669, 5695, 1111, 1160, 102], [101, 2277, 1105, 1208, 1122, 111, 108, 5347, 1580, 102], [101, 132, 188, 1103, 1322, 1104, 1103, 1503, 1989, 102], [101, 118, 146, 1253, 1138, 3828, 3058, 12398, 119, 102], [101, 1109, 3112, 1334, 1110, 1115, 146, 1238, 111, 102], [101, 108, 5347, 1580, 132, 189, 1138, 1251, 1168, 102], [101, 1334, 3154, 119, 1109, 1911, 1104, 1217, 1669, 102], [101, 1714, 1108, 1177, 27386, 119, 119, 119, 2586, 102], [101, 2225, 119, 107, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], 'overflow_to_sample_mapping': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}
"""
```

```python
# 把数据集转换为 Pandas
drug_dataset.set_format("pandas")

drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# 将默认的 "test" 部分重命名为 "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# 将 "test" 部分添加到我们的 `DatasetDict` 中
drug_dataset_clean["test"] = drug_dataset["test"]

#以 Arrow 格式保存我们清洗过的数据集
drug_dataset_clean.save_to_disk("drug-reviews")
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")

for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
    
# 检查当前进程的内存使用情况
import psutil

# Process.memory_info 是以字节为单位的,所以转换为兆字节
print(f"使用的RAM:{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
```

``` python
data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"

# 流式数据
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
# streaming=True 返回的对象是一个 IterableDataset 。顾名思义，要访问 IterableDataset ，我们需要迭代它
next(iter(pubmed_dataset_streamed))

shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10000, seed=42)
next(iter(shuffled_dataset))
# 从缓冲区的前 10,000 个示例中随机选择了一个示例。一旦访问了一个示例，它在缓冲区中的位置就会被语料库中的下一个示例填充 （即，上述案例中的第 10,001 个示例）。你还可以使用 IterableDataset.take() 和 IterableDataset.skip() 函数从流式数据集中选择元素，它的作用类似于 Dataset.select() 。

# 跳过前 1,000 个示例 ,将其余部分创建为训练集
train_dataset = shuffled_dataset.skip(1000)
# 将前 1,000 个示例用于验证集
validation_dataset = shuffled_dataset.take(1000)

law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
# interleave_datasets() 函数，它将一个 IterableDataset 对象列表组合为单个的 IterableDataset ，其中新数据集的元素是交替抽取列表中的数据集获得的。
from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
#  islice() 函数从合并的数据集中选择前两个示例
list(islice(combined_dataset, 2))
```

``` python
from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)

from datasets import Dataset
comments_dataset = Dataset.from_pandas(comments_df)

comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)

from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

import torch
device = torch.device("cuda")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
```

``` python
embeddings_dataset.add_faiss_index(column="embeddings")
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()

scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
```

### Tokenizers

大多数 Transformer 模型使用 `子词分词算法` 。为了找到语料库中的常见子词，tokenizer 需要深入统计语料库中的所有文本——这个过程我们称之为 `训练 （training）` 。

``` python
from datasets import load_dataset

# 加载这个可能需要几分钟的时间,你可以趁此喝杯咖啡或茶!
raw_datasets = load_dataset("code_search_net", "python")

# 生成器
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )
 # OR   
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]

training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokens = tokenizer.tokenize(example)

tokenizer.save_pretrained("code-search-net-tokenizer")
```

```python
# tokenizer 的输出不是简单的 Python 字典；我们得到的实际上是一个特殊的 BatchEncoding 对象。它是字典的子类（这就是为什么我们之前能够直接使用索引获取结果的原因）
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
# 直接得到Ttokenization 之前的单词而无需将 ID 转换回单词
encoding.tokens()
# 使用 word_ids() 方法来获取每个 token 原始单词的索引
encoding.word_ids()
```

![](D:/workfileS/coding/gitfile/md/pictures/210.png)

#### 标准化

标准化步骤涉及一些常规清理，例如删除不必要的空格、小写和“/”或删除重音符号。

``` python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#tokenizer 对象的 normalizer 属性具有一个 normalize_str() 方法
#bert-base-uncased checkpoint，在标准化的过程中转换为小写并删除重音。
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
```

#### 预分词

tokenizer 一般不会在原始文本上进行训练。因此，我们首先需要将文本拆分为更小的实体，例如单词。这就是预分词步骤的作用。基于单词的 tokenizer 可以简单地根据空格和标点符号将原始文本拆分为单词。这些词将是 tokenizer 在训练期间可以学习的子词的边界。

``` python
#查看快速 tokenizer 如何执行预分词，我们可以使用 `tokenizer` 对象的 `pre_tokenizer` 属性的 `pre_tokenize_str()` 方法
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

| 模型     | BPE                                              | WordPiece                                                    | Unigram                                              |
| -------- | ------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| 训练     | 从小型词汇表开始，学习合并 token 的规则          | 从小型词汇表开始，学习合并 token 的规则                      | 从大型词汇表开始，学习删除 token 的规则              |
| 训练步骤 | 合并对应最常见的 token 对                        | 合并对应得分最高的 token 对，优先考虑每个独立 token 出现频率较低的对 | 删除会在整个语料库上最小化损失的词汇表中的所有 token |
| 学习     | 合并规则和词汇表                                 | 仅词汇表                                                     | 含有每个 token 分数的词汇表                          |
| 编码     | 将一个单词分割成字符并使用在训练过程中学到的合并 | 从开始处找到词汇表中的最长子词，然后对其余部分做同样的事     | 使用在训练中学到找到最可能的 token 分割方式          |

#### `BPE`算法

1. BPE 训练首先计算语料库中使用的唯一单词集合（在完成标准化和预分词步骤之后），然后取出用来编写这些词的所有符号来构建词汇表。在实际应用中，基本词汇表将至少包含所有 ASCII 字符，可能还包含一些 Unicode 字符。如果你正在 tokenization 不在训练语料库中的字符，则该字符将转换为未知 tokens

    > GPT-2 和 RoBERTa （这两者非常相似）的 tokenizer 有一个巧妙的方法来处理这个问题：他们不把单词看成是用 Unicode 字符编写的，而是用字节编写的。这样，基本词汇表的大小很小（256），但是能包含几乎所有你能想象的字符，而不会最终转换为未知 tokens 这个技巧被称为 `字节级（byte-level） BPE` 。

    ```shell
    "hug", "pug", "pun", "bun", "hugs"
    # 基础单词集合将是 ["b", "g", "h", "n", "p", "s", "u"] 
    ```

2. 获得这个基础单词集合后，我们通过学习 `合并merges` 来添加新的 tokens 直到达到期望的词汇表大小。合并是将现有词汇表中的两个元素合并为一个新元素的规则。所以，一开始会创建出含有两个字符的 tokens 然后，随着训练的进展，会产生更长的子词。

    > 在分词器训练期间的任何一步，BPE 算法都会搜索最常见的现有 tokens 对 （在这里，“对”是指一个词中的两个连续 tokens ）。最常见的这一对会被合并，然后我们重复这个过程。

    ```shell
    ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
    #1. 通过将每个单词拆分为字符（形成我们初始词汇表的字符）来开始训练，这样我们就可以将每个单词视为一个 tokens 列表
    ("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
    # 最常见的对属于 ("u", "g") ，它在 "hug" 、 "pug" 和 "hugs" 中出现，总共在词汇表中出现了 20 次。
    # tokenizer 学习的第一个合并规则是 ("u", "g") -> "ug" ，意思就是 "ug" 将被添加到词汇表中，且应在语料库的所有词中合并这一对。
    词汇表: ["b", "g", "h", "n", "p", "s", "u", "ug"]
    语料库: ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
    # 这个阶段出现频率最高的对是 ("u", "n") ，在语料库中出现 16 次，所以学到的第二个合并规则是 ("u", "n") -> "un" 。
    词汇表: ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]
    语料库: ("h" "ug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("h" "ug" "s", 5)
    ```

    完成训练之后就可以对新的输入 tokenization 了，从某种意义上说，新的输入会依照以下步骤对新输入进行 tokenization：标准化、预分词、将单词拆分为单个字符、根据学习的合并规则，按顺序合并拆分的字符。Tokenizer 学习到了三个合并规则：

    ``` shell
    ("u", "g") -> "ug"
    ("u", "n") -> "un"
    ("h", "ug") -> "hug"
    ```

    在这种情况下，单词 `"bug"` 将被转化为 `["b", "ug"]` 。然而 `"mug"` ，将被转换为 `["[UNK]", "ug"]` ，因为字母 `"m"` 不再基本词汇表中。同样，单词 `"thug"` 会被转换为 `["[UNK]", "hug"]` ：字母 `"t"` 不在基本词汇表中，使用合并规则首先会将 `"u"` 和 `"g"` 合并，然后将 `"h"` 和 `"ug"` 合并。

    ``` python
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    
    from transformers import AutoTokenizer
    from collections import defaultdict
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    word_freqs = defaultdict(int)
    
    # 进行预分词的同时计算语料库中每个单词的频率
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1
           
    # 计算基础词汇表，这由语料库中使用的所有字符组成
    alphabet = []
    
    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    # 在该词汇表的开头添加了模型使用的特殊 tokens 对于 GPT-2，唯一的特殊 tokens 是 "<|endoftext|>" ：
    vocab = ["<|endoftext|>"] + alphabet.copy()
    # 将每个单词拆分为单独的字符
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    # 计算每对字符的频率。
    def compute_pair_freqs(splits):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(a, b, splits):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
    
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    vocab_size = 50
    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])
    ```

3. 为了对新文本进行分词，我们对其进行预分词、拆分，然后使用学到的所有合并规则

    > 如果存在未知字符，我们的实现将抛出错误，因为我们没有做任何处理它们。GPT-2 实际上没有未知 tokens （使用字节级 BPE 时不可能得到未知字符），但这里的代码可能会出现这个错误，因为我们并未在初始词汇中包含所有可能的字节。

    ``` python
    def tokenize(text):
        pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]
        for pair, merge in merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
    
        return sum(splits, [])
    tokenize("This is not a token.")
    ```

#### `WordPiece`算法

与BPE 一样，WordPiece 也是从包含模型使用的特殊 tokens 和初始字母表的小词汇表开始的。由于它是通过添加前缀（如 BERT 中的 `##` ）来识别子词的，每个词最初都会通过在词内部所有字符前添加该前缀进行分割。因此，例如 `"word"` 将被这样分割：`w ##o ##r ##d`。因此，初始字母表包含所有出现在单词第一个位置的字符，以及出现在单词内部并带有 WordPiece 前缀的字符。

然后，同样像 BPE 一样，WordPiece 会学习合并规则。主要的不同之处在于合并对的选择方式。WordPiece 不是选择频率最高的对，而是对每对计算一个得分，使用以下公式
$$
\text{score}(t_1, t_2) = \frac{\text{freq}(t_{1, 2})}{\text{freq}(t_1)\times\text{freq}(t_2)}
$$
其中：$t_1$和$t_2$为两个 `token`，$t_{1,2}$为它们 `merge` 之后得到的新的 `token` 。$\text{freq}(t)$为 `token` 在语料库中出现的频次。通过将两部分合在一起的频率除以其中各部分的频率的乘积，该算法优先合并那些在词汇表中单独出现出现的对。

``` shell
# 分割
("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##g" "##s", 5)
词汇表: ["b", "h", "p", "##g", "##n", "##s", "##u"] 
# 分数最高的对是 ("##g", "##s") —— 唯一没有 "##u" 的对——分数是 1 / 20，所以学习的第一个合并是 ("##g", "##s") -> ("##gs") 。
# 当我们合并时，我们会删除两个 tokens 之间的 ## ，所以我们将 "##gs" 添加到词汇表中，并将语料库的单词按照改规则进行合并
词汇表: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs"]
语料库: ("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("h" "##u" "##gs", 5)

# 此时， "##u" 出现在所有可能的对中，因此它们最终都具有相同的分数。在这种情况下，第一个对会被合并，于是我们得到了 ("h", "##u") -> "hu" 规则
词汇表: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu"]
语料库: ("hu" "##g", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)

# 下一个最佳得分的对是 ("hu", "##g") 和 ("hu", "##gs") （得分为 1/15，而所有其他配对的得分为 1/21）
词汇表: ["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu", "hug"]
语料库: ("hug", 10), ("p" "##u" "##g", 5), ("p" "##u" "##n", 12), ("b" "##u" "##n", 4), ("hu" "##gs", 5)
```

``` python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# 然后我们在进行预分词的同时，计算语料库中每个单词的频率 参考BPE

# 字母表是一个独特的集合，由所有单词的第一个字母以及所有以 ## 为前缀和在单词中的其他字母组成
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
# 我们还在该词汇表的开头添加了模型使用的特殊 tokens，在使用 BERT 的情况下，特殊 tokens 是 ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}

def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

vocab_size = 70
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)
```

要对新文本进行分词，我们先预分词，再进行分割，然后在每个词上使用分词算法。也就是说，我们寻找从第一个词开始的最大子词并将其分割

``` python
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens

def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])

tokenize("This is the Hugging Face course!")
```

#### `Unigram`算法

与BPE 和 WordPiece 相比，Unigram 的工作方式正好相反：它从一个大词汇库开始，然后逐步删除词汇，直到达到目标词汇库大小。

在训练的每一步，Unigram 算法都会在给定当前词汇的情况下计算语料库的损失。然后，对于词汇表中的每个符号，算法计算如果删除该符号，整体损失会增加多少，并寻找删除后损失增加最少的符号。

> 这个过程非常消耗计算资源，因此我们不只是删除与最低损失增长相关的单个符号，而是删除与最低损失增长相关的百分之p （p 是一个可以控制的超参数，通常是 10 或 20）的符号。然后重复此过程，直到词汇库达到所需大小。注意，我们永远不会删除基础的单个字符，以确保任何词都能被切分。

``` python
词汇表: ["h", "u", "g", "hu", "ug", "p", "pu", "n", "un", "b", "bu", "s", "hug", "gs", "ugs"]
```

Unigram 模型是一种语言模型，它认为每个符号都与其之前的符号独立。这是最简单的语言模型，因此给定之前的上下文情况下，符号 X 的概率就是符号 X 的概率。所以，如果我们使用 Unigram 语言模型来生成文本，我们的预测总会输出最常见的符号。给定符号的概率是其在原始语料库中的频率（我们计算它出现的次数），除以词汇库中所有符号的频率总和（以确保概率总和为 1）。

现在，为了对一个给定的单词进行分词，我们会查看所有可能的分词组合，并根据 Unigram 模型计算出每种可能的概率。由于所有的分词都被视为独立的，因此这个单词分词的概率就是每个子词概率的乘积。

``` python
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
# 我们需要将我们的词汇表初始化为大于我们最终想要的词汇量。我们必须包含所有基本的单个字符（否则我们将无法对每个单词赋予一个 token ），但对于较大的子字符串，我们将只保留最常见的字符，因此我们按出现频率对它们进行排序
char_freqs = defaultdict(int)
subwords_freqs = defaultdict(int)
for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # 循环遍历长度至少为2的子字
        for j in range(i + 2, len(word) + 1):
            subwords_freqs[word[i:j]] += freq

# 按频率对子词排序
sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
token_freqs = list(char_freqs.items()) + sorted_subwords[: 300 - len(char_freqs)]
token_freqs = {token: freq for token, freq in token_freqs}

from math import log

total_sum = sum([freq for token, freq in token_freqs.items()])
# 我们需要计算所有频率的总和，将频率转化为概率。在我们的模型中，我们将存储概率的对数，因为相较于小数相乘，对数相加在数值上更稳定，而且这将简化模型损失的计算
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}


def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # best_score_at_start应该由循环的前面的步骤计算和填充
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # 如果我们发现以 end_idx 结尾的更好分段,我们会更新
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # 我们没有找到单词的 tokens  -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score

def compute_loss(model):
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        loss += freq * word_loss
    return loss

import copy

def compute_scores(model):
    scores = {}
    model_loss = compute_loss(model)
    for token, score in model.items():
        # 我们将保留长度为 1 的 tokens
        if len(token) == 1:
            continue
        model_without_token = copy.deepcopy(model)
        _ = model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token) - model_loss
    return scores

percent_to_remove = 0.1
while len(model) > 100:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    # 删除分数最低的percent_to_remov tokens 。
    for i in range(int(len(model) * percent_to_remove)):
        _ = token_freqs.pop(sorted_scores[i][0])

    total_sum = sum([freq for token, freq in token_freqs.items()])
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
    
def tokenize(text, model):
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in words_with_offsets]
    encoded_words = [encode_word(word, model)[0] for word in pre_tokenized_text]
    return sum(encoded_words, [])


tokenize("This is the Hugging Face course.", model)
```

#### 模块化构建tokenizer

``` python
# 获取语料库
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]
    
    
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# tokenization 的第一步是标准化，所以我们从这里开始。由于 BERT 被广泛使用，所以我们可以使用 BertNormalizer ，我们可以为 BERT 设置经典参数： lowercase（小写） 和 strip_accents（去除重音的字符） ， clean_text 用于删除所有控制字符并将重复的空格替换为一个空格；以及 handle_chinese_chars ，它将在中文字符周围添加空格。
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# 手动创建 BERT normalizer 。
#我们还使用了一个 NFD Unicode normalizer ，否则，否则 StripAccents normalizer 将因为无法正确识别带有重音的字符，从而没办法去除重音。
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
# 可以使用 normalizer 的 normalize_str() 方法来对它进行测试
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

# 下一步是预分词。同样，我们可以使用预构建的 BertPreTokenizer
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# OR
#  Whitespace 会使用空格和所有不是字母、数字或下划线的字符进行分割
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# tokenization 流程的下一步是将输入数据传递给模型。我们已经在初始化时指定了我们的模型，但是我们还需要对其进行训练，这就需要一个 WordPieceTrainer 。
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# 除了指定 vocab_size 和 special_tokens ，我们还可以设置 min_frequency （一个 tokens 必须达到的最小的出现的次数才能被包含在词汇表中）或更改 continuing_subword_prefix （如果我们想使用其他的字符来替代 ## ）。
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# 我们还可以使用本地的文本文件来训练我们的 tokenizer 
tokenizer.model = models.WordPiece(unk_token="[UNK]")
tokenizer.train(["wikitext-2.txt"], trainer=trainer)

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
# 所得到的 encoding 是一个 Encoding 对象，它包含 tokenizer 的所有必要属性： ids 、 type_ids 、 tokens 、 offsets 、 attention_mask 、 special_tokens_mask 和 overflowing 。

#tokenizer 管道的最后一步是后处理。我们需要在开头添加 [CLS] token，然后在结束时（或在每句话后，如果我们有一对句子）添加 [SEP] token。我们将使用 TemplateProcessor 来完成这个任务，但首先我们需要知道词汇表中 [CLS] 和 [SEP] tokens 的 ID：
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")

# 编写 TemplateProcessor 的模板时，我们必须指定如何处理单个句子和一对句子。对于这两者，我们写下我们想使用的特殊 tokens 第一句（或单句）用 $A 表示，而第二句（如果需要编码一对句子）用 $B 表示。对于这些（特殊 tokens 和句子），我们还需要在冒号后指定相应的 token 类型 ID。
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# 最后一步：指定一个解码器
tokenizer.decoder = decoders.WordPiece(prefix="##")
tokenizer.decode(encoding.ids)  # 测试

tokenizer.save("tokenizer.json")

new_tokenizer = Tokenizer.from_file("tokenizer.json")

# 要将构建的 tokenizer 封装在 PreTrainedTokenizerFast 类中，我们可以将我们构建的 tokenizer 作为 tokenizer_object 传入，或者将我们保存的 tokenizer 文件作为 tokenizer_file 传入。要记住的关键一点是，我们需要手动设置所有的特殊 tokens，因为这个类不能从 tokenizer 对象推断出哪个符号是掩码符号， [CLS] 符号等
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # 也可以从tokenizer文件中加载
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
# 可以使用 save_pretrained() 方法来保存它，或者使用 push_to_hub() 方法将它上传到 Hub。
```

``` python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# trim_offsets = False 这个选项告诉 post-processor，我们应该让那些以‘Ġ’开始的 tokens 的偏移量保持不变：这样，偏移量起始的索引将指向单词前的空格，而不是单词的第一个字符（因为空格在技术上是 token 的一部分）。
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)
```

``` python
tokenizer = Tokenizer(models.Unigram())
from tokenizers import Regex

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
cls_token_id = tokenizer.token_to_id("<cls>")
sep_token_id = tokenizer.token_to_id("<sep>")
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)
tokenizer.decoder = decoders.Metaspace()

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)
```

#### `NLP`任务

首先探讨的应用是 Token 分类。这个通用任务涵盖了所有可以表述为“给句子中的词或字贴上标签”的问题，例如：

- 实体命名识别 `NER`：找出句子中的实体（如人物、地点或组织）。这可以通过为每个实体指定一个类别的标签，如果没有实体则会输出无实体的标签。
- 词性标注 `POS`：将句子中的每个单词标记为对应于特定的词性（如名词、动词、形容词等）。
- 分块`chunking`：找出属于同一实体的 tokens 这个任务（可以与词性标注或命名实体识别结合）可以被描述为将位于块开头的 token 赋予一个标签（通常是 “ `B-` ” （Begin），代表该token位于实体的开头），将位于块内的 tokens 赋予另一个标签（通常是 “ `I-` ”（inner）代表该token位于实体的内部），将不属于任何块的 tokens 赋予第三个标签（通常是 “ `O` ” （outer）代表该token不属于任何实体）。



### 模型

![](D:/workfileS/coding/gitfile/md/pictures/208.png)

我们使用 tokenizer，它将负责：

- 将输入拆分为单词、子单词或符号（如标点符号），称为 **token**（标记）
- 将每个标记（token）映射到一个数字，称为 **input ID**（inputs ID）
- 添加模型需要的其他输入，例如特殊标记（如 `[CLS]` 和 `[SEP]` ）
- - 位置编码：指示每个标记在句子中的位置。
- - 段落标记：区分不同段落的文本。
- - 特殊标记：例如 [CLS] 和 [SEP] 标记，用于标识句子的开头和结尾。

``` python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}

model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
torch.Size([2, 16, 768])

# 以情感分类为例，我们需要一个带有序列分类头的模型
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

Transformers 模型的输出会直接发送到模型头进行处理。模型头通常由一个或几个线性层组成，它的输入是隐状态的高维向量，它会并将其投影到不同的维度。

![](D:/workfileS/coding/gitfile/md/pictures/209.png)

### Tokenizer

``` python
from transformers import AutoTokenizer
#AutoTokenizer 类将根据 checkpoint 名称在库中获取正确的 tokenizer 类，并且可以直接与任何 checkpoint 一起使用
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

将文本翻译成数字被称为编码`encoding`。编码分两步完成：分词，然后转换为 inputs ID。

- 第一步是将文本拆分为单词（或部分单词、标点符号等），通常称为 tokens 不同的分词器使用的算法也不一样。

- 第二步是将这些 tokens 转换为数字，可以用它们构建一个张量并将它们提供给模型。为此，tokenizer 有一个词汇表`vocabulary`

``` python
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)  # ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)  # [7993, 170, 11303, 1200, 2443, 1110, 3014]

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)  # 'Using a Transformer network is simple'
```

Padding 通过在值较少的句子中添加一个名为 `padding_id` 的特殊单词来确保我们所有的句子长度相同。在 `tokenizer.pad_token_id` 中找到填充 token 的 ID。当我们使用填充`padding`来处理长度不同的句子时，我们会添加特殊的“填充 token”来使所有句子达到相同的长度。但是，注意力层会将这些填充 token 也纳入考虑，因为它们会关注序列中的所有 tokens。这就导致了一个问题：尽管填充 token 本身并没有实际的含义，但它们的存在会影响模型对句子的理解。我们需要告诉这些注意层忽略填充 token。这是通过使用注意力掩码`attention mask`层来实现的。

注意力掩码`attention mask`是与 inputs ID 张量形状完全相同的张量，用 0 和 1 填充：1 表示应关注相应的 tokens，0 表示应忽略相应的 tokens

``` python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
```

`token类型ID(token_type_ids)` 的作用就是告诉模型输入的哪一部分是第一句，哪一部分是第二句。

```python
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer("This is the first sentence.", "This is the second one.")

{
    "input_ids": [
        101,
        2023,
        2003,
        1996,
        2034,
        6251,
        1012,
        102,
        2023,
        2003,
        1996,
        2117,
        2028,
        1012,
        102,
    ],
    "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}
```

#### 微调大语言模型

``` python
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 和之前一样
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")
"""DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})"""

#这种方法虽然有效，但有一个缺点是它返回的是一个字典（字典的键是 输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 token类型ID(token_type_ids) ，字典的值是键所对应值的列表）。
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)

#该函数接收一个字典（与 dataset 的项类似）并返回一个包含 输入词id(input_ids) ， 注意力遮罩(attention_mask) 和 token_type_ids 键的新字典。
#如果 example 字典所对应的值包含多个句子（每个键作为一个句子列表），那么它依然可以运行, tokenizer 可以处理成对的句子列表，这样的话我们可以在调用 map() 时使用该选项 batched=True ，这将显著加快处理的速度。 
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

#我们暂时在 tokenize_function 中省略了 padding 参数。这是因为将所有的样本填充到最大长度有些浪费。一个更好的做法是：在构建 batch 的时候。这样我们只需要填充到每个 batch 中的最大长度，而不是整个数据集的最大长度。
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# Datasets 库进行这种处理的方式是向数据集添加新的字段，每个字段对应预处理函数返回的字典中的每个键
"""DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})"""
# 如果预处理函数 map() 为现有键返回一个新值，我们可以通过使用 map() 函数返回的新值修改现有的字段。
```

为了解决句子长度不统一的问题，我们必须定义一个 collate 函数，该函数会将每个 batch 句子填充到正确的长度

``` python
from transformers import DataCollatorWithPadding
# 需要一个 tokenizer （用来知道使用哪种填充 token 以及模型期望在输入的左边填充还是右边填充）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]

# [50, 59, 47, 67, 59, 50, 62, 32]
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
"""
{
    "attention_mask": torch.Size([8, 67]),
    "input_ids": torch.Size([8, 67]),
    "token_type_ids": torch.Size([8, 67]),
    "labels": torch.Size([8]),
}
"""
```

``` python
## training

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification
# 在实例化此预训练模型后会收到警告。这是因为 BERT 没有在句子对分类方面进行过预训练，所以预训练模型的 head 已经被丢弃，而是添加了一个适合句子序列分类的新头部。这些警告表明一些权重没有使用（对应于被放弃的预训练头的权重），而有些权重被随机初始化（对应于新 head 的权重）。
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
```

``` python
# 构建一个有用的 compute_metrics() 函数，并在下次训练时使用它。该函数必须接收一个 EvalPrediction 对象（它是一个带有 predictions 和 label_ids 字段的参数元组），并将返回一个字符串映射到浮点数的字典（字符串是返回的指标名称，而浮点数是其值）。
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
# (408, 2) (408,)
# predict() 方法的输出另一个带有三个字段的命名元组: predictions label_ids 和 metrics metrics 字段将只包含所传递的数据集的损失,以及一些时间指标(总共花费的时间和平均预测时间)。当我们定义了自己的 compute_metrics() 函数并将其传递给 Trainer 该字段还将包含 compute_metrics() 返回的结果。
import evaluate
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```



``` python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练前准备
# 删除与模型不需要的列（如 sentence1 和 sentence2 列）。
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# 将列名 label 重命名为 labels （因为模型默认的输入是 labels ）。
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 设置数据集的格式，使其返回 PyTorch 张量而不是列表。
tokenized_datasets.set_format("torch")

# 定义数据加载器
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# 为了确保训练过程中一切顺利，我们将 batch 传递给这个模型
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

#优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
# 默认使用的学习率调度器只是从最大值 （5e-5） 到 0 的线性衰减。为了定义它，我们需要知道我们训练的次数，即所有数据训练的次数（epochs）乘以的 batch 的数量（即我们训练数据加载器的长度）。 
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 训练循环
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps)) 
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

``` python
# 通过使用accelerate在多个 GPU 或 TPU 上启用分布式训练。
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

把这个放在 `train.py` 文件中，可以让它在任何类型的分布式设置上运行。要在分布式设置中试用它，请运行以下命令：

```
accelerate config
```

```
accelerate launch train.py
```

如果你想在 Notebook 中尝试此操作（例如，在 Colab 上使用 TPU 进行测试），只需将代码粘贴到一个 `training_function()` 函数中，并在最后一个单元格中运行：

```
from accelerate import notebook_launcher
notebook_launcher(training_function)
```

