import torch
from transformers import pipeline, set_seed
import os
import pandas as pd
import argparse
import time
from functools import partial
import requests
import openai
from openai import OpenAI
client = OpenAI(api_key='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')


def generate_with_a_prompt_gpt2(prompt, text_gen_pipeline, n=500, pad_token_id=50256):
    """
    Generate a list of statements given the prompt based on one GPT-2 model

    NOTE: 50256 corresponds to '<|endoftext|>'
    """

    results = text_gen_pipeline(prompt,
                                do_sample=True,
                                max_length=50,
                                temperature=1.0,
                                num_return_sequences=n,  # 1000 leads to OOM for original gpt2
                                pad_token_id=pad_token_id,
                                clean_up_tokenization_spaces=True
                                )

    # only use the first utterance
    results = [res['generated_text'].split("\n")[0] for res in results]
    return results


def generate_with_a_prompt_gpt3(prompt, max_tokens=50, n=500, temperature=1.):
    n1 = 0
    n2 = 0
    response = 'none'
    while True:
        try:
            if n1 >= 5:
                print(f'{prompt}, too many times wrong label')
                break
            if n2 >= 5:
                print(f'{prompt}, too many requested time out errors')
                break

            completion = client.completions.create(
                prompt=prompt,
                model='text-ada-001',
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
                timeout=5
            )
            response = completion.choices[0].text
            break
        except openai.RateLimitError:
            print("Rate Limit Error")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            print("Request Time Out")
            time.sleep(2)
            n2 += 1
        except KeyboardInterrupt:
            time.sleep(2)
        except Exception as e:
            print(e, prompt)
            break

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../02_model_finetuning/models/gpt2_community_0')
    parser.add_argument("--prompt_data_path", type=str, default='../data/anes2020_pilot_prompt_probing.csv')
    parser.add_argument("--preceding_prompt", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument('--max_n_tokens', type=int, default=50)
    parser.add_argument('--n_responses', type=int, default=500)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(args.model_path)
    print(device)

    os.makedirs(args.output_dir, exist_ok=True)

    df_prompt = pd.read_csv(args.prompt_data_path)
    # df_prompt = df_prompt.iloc[:2]
    prompts1 = df_prompt['Prompt1'].tolist()
    prompts2 = df_prompt['Prompt2'].tolist()
    prompts3 = df_prompt['Prompt3'].tolist()
    prompts4 = df_prompt['Prompt4'].tolist()
    pids = df_prompt['pid'].tolist()

    if 'gpt2' in args.model_path:
        if 'community' in args.model_path:
            model_path = args.model_path
        else:
            model_path = 'gpt2'
        text_generator = pipeline('text-generation', model=model_path, device=device)
        generate_with_a_prompt = partial(generate_with_a_prompt_gpt2, text_gen_pipeline=text_generator, n=args.n_responses, pad_token_id=50256)
    else:
        generate_with_a_prompt = partial(generate_with_a_prompt_gpt3, max_token=50, n=args.n_responses, temperature=1.)

    data = []
    for seed in range(args.n_runs):
        set_seed(seed)
        for prompts in [prompts1, prompts2, prompts3, prompts4]:
            for pid, prompt in zip(pids, prompts):
                print(f"seed: {seed}, pid: {pid}, prompt: {prompt}")
                if args.preceding_prompt:
                    full_prompt = " ".join([args.preceding_prompt, prompt])
                else:
                    full_prompt = prompt
                batch_responses = generate_with_a_prompt(prompt=full_prompt)
                data.extend([[seed, pid, prompt, response.replace(args.preceding_prompt, '')] for response in batch_responses])

    df_responses = pd.DataFrame(data, columns=['seed', 'pid', 'prompt', 'response'])

    def process_responses(s):
        s = s.replace("\n", " ")
        s = s.strip()
        return s

    df_responses['response'] = df_responses['response'].apply(process_responses)
    print(f"# of responses: {df_responses.shape[0]}, # of unique responses: {df_responses['response'].nunique()}")

    model_name = os.path.basename(args.model_path)
    output_path = f'{args.output_dir}/{model_name}.csv'
    df_responses.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()