import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

BOS = "<|endoftext|>"
EOM = "<|endofmask|>"
INFILL = "<infill>"
FILE = "<|/ file |>"


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def make_sentinel(i: int):
    return f"<|mask:{i}|>"


def generate(
    model,
    tokenizer: PreTrainedTokenizerBase,
    input: str,
    max_to_generate: int = 128,
    temperature: float = 0.2,
):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    """
    input_ids = tokenizer(input, return_tensors="pt").input_ids.cuda()
    max_length = max_to_generate + input_ids.flatten().size(0)
    # if max_length > 2048:
    #     print(
    #         "warning: max_length {} is greater than the context window {}".format(
    #             max_length, 2048
    #         )
    #     )
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.95,
            temperature=temperature,
            max_length=max_length,
        )
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(
        output.flatten(), clean_up_tokenization_spaces=False
    )
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS) :]
    return detok_hypo_str


def infill(
    model,
    tokenizer: PreTrainedTokenizerBase,
    parts: list[str],
    max_to_generate: int = 128,
    temperature: float = 0.2,
    extra_sentinel: bool = True,
    max_retries: int = 1,
):
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    prompt = ""
    infills = []
    text = ""
    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        # print(f"retry {retries_attempted}")

        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)

        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            completion = generate(
                model, tokenizer, prompt, max_to_generate, temperature
            )
            completion = completion[len(prompt) :]
            if EOM not in completion:
                # print(f"warning: {EOM} not found")
                completion += EOM
                done = False
            completion = completion[: completion.index(EOM) + len(EOM)]
            infilled = completion[: -len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = "".join(complete)

    return {
        "text": text,  # str, the completed document (with infills inserted)
        "parts": parts,  # List[str], length N. Same as passed to the method
        "infills": infills,  # List[str], length N-1. The list of infills generated
        "retries_attempted": retries_attempted,  # number of retries used (if max_retries > 1)
    }
