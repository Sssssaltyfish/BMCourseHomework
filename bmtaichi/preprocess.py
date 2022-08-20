import json
from pathlib import Path

scatter_word_list = [
    "scatter",
    "'scatter'",
    '"scatter"',
    "scatter_kws",
    "'o'",
    "'bo'",
    "'r+'",
    '"o"',
    '"bo"',
    '"r+"',
]
hist_word_list = [
    "hist",
    "'hist'",
    '"hist"',
    "bar",
    "'bar'",
    '"bar"',
    "countplot",
    "barplot",
]
pie_word_list = ["pie", "'pie'", '"pie"']
scatter_plot_word_list = ["lmplot", "regplot"]
hist_plot_word_list = ["distplot", "kdeplot", "contour"]
normal_plot_word_list = ["plot"]

reserved_words = (
    scatter_word_list
    + hist_word_list
    + pie_word_list
    + scatter_plot_word_list
    + hist_plot_word_list
    + normal_plot_word_list
)


def preprocess_file(filepath: Path, is_train=True):
    plot_samples = []
    clean_samples = []

    with filepath.open("r") as fin:
        for i, line in enumerate(fin):
            sample = json.loads(line)

            init_code_seq = sample["code_tokens"]
            code_seq = [
                tok for tok in init_code_seq if not (len(tok) == 0 or tok[0] == "#")
            ]

            TARGET_TOKEN = "plt"

            while TARGET_TOKEN in code_seq:
                pos = code_seq.index(TARGET_TOKEN)
                if pos < len(code_seq) - 1 and code_seq[pos + 1] == ".":
                    break
                code_seq = code_seq[pos + 1 :]

            if TARGET_TOKEN not in code_seq:
                continue

            plot_calls = []
            api_seq = sample["api_sequence"]
            for api in api_seq:
                if api == "subplot":
                    continue
                if api[-4:] == "plot" and not ("_" in api):
                    plot_calls.append(api)

            exist_plot_calls = False
            for code_idx, tok in enumerate(code_seq):
                if not (tok in reserved_words + plot_calls):
                    continue
                if code_idx == len(code_seq) - 1 or code_seq[code_idx + 1] != "(":
                    continue
                exist_plot_calls = True
                break
            if not exist_plot_calls:
                continue

            url = sample["metadata"]["path"]
            if "solution" in url.lower() or "assignment" in url.lower():
                clean_samples.append(sample)
                if not is_train:
                    plot_samples.append(sample)
            else:
                plot_samples.append(sample)

    print("number of samples in the original partition: ", len(plot_samples))
    print("number of course-related samples in the partition: ", len(clean_samples))
    return plot_samples, clean_samples


if __name__ == "__main__":
    directory = Path("data")
    SUFFIX = "_preprocessed"
    for file in directory.iterdir():
        print(f"preprocessing {file}")
        stem = file.stem
        if file.is_file() and not stem.endswith(SUFFIX):
            if "train" in stem:
                plot_samples, clean_samples = preprocess_file(file)
            else:
                plot_samples, clean_samples = preprocess_file(file, False)

            with file.with_stem(stem + SUFFIX).open("w") as fout:
                json.dump(plot_samples, fout)
