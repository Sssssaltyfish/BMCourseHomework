from inspect import signature
from typing import Any, Callable, Optional, TypeVar
from bminf.models import CPM1, CPM2, EVA

import argparse


def get_commandline_args():
    parser = argparse.ArgumentParser(description="Text Generation Demo")
    parser.add_argument("task", choices=["fill", "story", "dialogue"])
    args = parser.parse_args()
    return args


def load_model(choice: str):
    return {
        "fill": CPM2,
        "story": CPM1,
        "dialogue": EVA,
    }[choice]()


def interactive(
    task: Callable[[], None],
    before_task: Optional[Callable[[], None]] = None,
    after_task: Optional[Callable[[], None]] = None,
):
    if before_task is None:
        before_task = lambda: None
    if after_task is None:
        after_task = lambda: print("\nExited.")

    before_task()
    try:
        while True:
            task()
    except KeyboardInterrupt:
        after_task()


def gen_fill(model: CPM2, text: str):
    TOKEN_SPAN = "<span>"
    text = text.replace("__", TOKEN_SPAN)
    args = {
        "top_p": 1.0,
        "top_n": 10,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    result = model.fill_blank(text, **args)

    for v in result:
        value = v["text"]
        text = text.replace(TOKEN_SPAN, value, 1)
    return {"list": result, "resp": text}


def fill(model: CPM2):
    def fill_():
        inputs = input("> ")
        result = gen_fill(model, inputs)
        print("model:", result["resp"])

    interactive(
        fill_,
        lambda: print(
            "Input a sentence to be filled in, use '__' (double underscore) to represent blanks. "
        ),
    )


def gen_story(model: CPM1, text: list[str]):
    args = {
        "max_tokens": 128,
        "top_p": 1.0,
        "top_n": 10,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    result = [model.generate(' '.join(text), **args) for _ in range(len(text))]
    return result


def story(model: CPM1):
    def story_():
        inputs = input("> ")
        result = gen_story(model, inputs.split())

        text = ""
        for s, end in result:
            text += s
            if end: break
        print(f"Story generated: {inputs + text}")

    interactive(story_, lambda: print("Input some words to generate a story."))


def gen_dialogue(model: EVA, text: list[str]):
    args = {
        "max_tokens": 128,
        "top_p": 1.0,
        "top_n": 10,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    result = model.dialogue(text, **args)
    return result


def dialogue(model: EVA):
    print("Feel free to chat with the model!")
    history = []

    def dialogue_():
        inputs = input("You: ")
        history.append(inputs)
        result = gen_dialogue(model, history)

        print(f"Model: {result[0]}")

    interactive(dialogue_)


def main():
    args = get_commandline_args()
    print("Loading model, please wait.")
    model = load_model(args.task)
    print("Successfully loaded.")

    task = {"fill": fill, "story": story, "dialogue": dialogue}[args.task]
    task(model)


if __name__ == "__main__":
    main()
