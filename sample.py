"""
OpenAI is blocked in the China mainland, so I choose the Alibaba models which are compatible with
 the OpenAI API for the task.


.env files contains:
HF_ENDPOINT=https://hf-mirror.com
DASHSCOPE_API_KEY=sk-xxxxxxxxx

load the dataset and save to local disk.
ds = datasets.load_dataset("lukaemon/bbh", "reasoning_about_colored_objects")
ds.save_to_disk("D:/huggingface/lukaemon/bbh")
"""
import argparse
import asyncio
import json
import os
import re
import sys

import datasets
import openai
from datasets import tqdm, get_dataset_config_names

def get_prompt_template():
    print("Enter a prompt:")
    user_input = sys.stdin.readline().strip()
    if not user_input:
        user_input = """Answer the question: """
        print(user_input)
    return user_input


async def get_llm_response_async(prompt, prompt_template, model):
    formatted_prompt = prompt_template + prompt
    print("formatted prompt:", formatted_prompt)

    client = openai.AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are an expert in resolving issues of counting and logical judgment. Answer the single choice question from the Content. Provide the correct answer in the format: The answer is: (A)."},
            {"role": "user",
             "content": "###Q: On the table, you see a set of items arranged in a row: a red crayon, a yellow paperclip, a magenta necklace, a grey textbook, and a silver cat toy. What is the color of the left-most item? Options: (A) red (B) orange (C) yellow (D) green (E) blue (F) brown (G) magenta (H) fuchsia (I) mauve (J) teal (K) turquoise (L) burgundy (M) silver (N) gold (O) black (P) grey (Q) purple (R) pink"},
            {"role": "assistant", "content": "The answer is: (A)."},
            {"role": "user", "content": formatted_prompt}
        ],
        max_tokens=400,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0.7,
    )
    print("llm response:")
    print(response)

    return response.choices[0].message.content.strip()


def evaluate(response, target):
    target_label = target.strip("() ")
    response_label = ""
    response_labels = re.findall(r'The answer is: \((.?)\)', response)
    if len(response_labels) > 0:
        response_label = response_labels[0]

    print("response: ", response)
    print("response_label first:", response_label)
    print("target:", target_label)

    return response_label == target_label

async def main(args):
    ds = datasets.load_from_disk("D:/huggingface/lukaemon/bbh")
    subset = ds["test"].shuffle(seed=args.seed).select(range(args.limit))
    print(ds)

    async def process_example(example):
        async with asyncio.Semaphore(args.concurrency):
            input_ = example["input"]
            target = example["target"]
            response = await get_llm_response_async(input_, get_prompt_template(), args.model)
            is_correct = evaluate(response, target)
            result = {
                'question': input_,
                'ground_truth': target,
                'model_response': response,
                'is_correct': is_correct
            }
            return result

    tasks = [process_example(example) for example in subset]
    evaluation_results = await tqdm.gather(*tasks, desc="Processing examples")

    print("result:")
    print(evaluation_results)

    # calculate the accuracy of the result.
    accuracy = 0
    for i in evaluation_results:
        if i["is_correct"]:
            accuracy += 1
    print(f"Accuracy:", accuracy / len(tasks))

    # save the result to a json file.
    report = {
        "accuracy": accuracy / len(tasks),
        "result": evaluation_results
    }
    with open('result.json', 'w+', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)


# TODO: calculate accuracy
# TODO: implement any other metric that you think might be useful

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--limit", type=int, default=1)
    # qwen1.5-1.8b-chat qwen2-0.5b-instruct qwen2-1.5b-instruct qwen2-7b-instruct qwen2-72b-instruct qwen-turbo-0624
    # qwen-plus-0723
    args.add_argument("--model", type=str, default="qwen2-7b-instruct")
    args.add_argument("--seed", type=int, default=40)
    args.add_argument("--concurrency", type=int, default=1)
    args = args.parse_args()

    asyncio.run(main(args))
