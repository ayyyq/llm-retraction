import os
import sys

import numpy as np

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(src_path)
sys.path.append(src_path)
import json
import h5py
import argparse
import torch

from utils.general_utils import *
from utils.patchscopes_utils import *
from transformers import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from tqdm import tqdm


def load(model_path):
    sos_tok = False
    torch_dtype = torch.float16

    mt = ModelAndTokenizer(
        model_path,
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
        attn_implementation='eager'
    )
    mt.model.eval()
    torch.set_grad_enabled(False)
    return mt


def main():
    args = parse_args()
    mt = load(args.model_path)

    # read data
    data = [json.loads(line) for line in open(args.input_filename)]
    print(f'Loaded {len(data)} instances')

    if args.debug:
        data = data[:10]

    if args.save_activations:
        if args.template == 'plain':
            max_prompt_len = max([len(mt.tokenizer.encode(instance['prompt'], add_special_tokens=False)) for instance in data])
        elif args.template == 'chat':
            max_prompt_len = max([len(mt.tokenizer.apply_chat_template(instance['messages'], add_generation_prompt=True, tokenize=True)) for instance in data])
        elif args.template == 'continuation':
            max_prompt_len = max([len(mt.tokenizer.apply_chat_template(instance['messages'], continue_final_message=True, tokenize=True)) for instance in data])
        else:
            raise NotImplementedError
        if args.fixed_token is not None:
            max_prompt_len += len(mt.tokenizer.encode(args.fixed_token, add_special_tokens=False))
        print(f'Max prompt length: {max_prompt_len}')

        num_tokens = 1
        if args.fixed_token is not None:
            num_tokens += len(mt.tokenizer.encode(args.fixed_token, add_special_tokens=False))
            # TODO: check
            assert num_tokens == 2
        attentions = torch.zeros(len(data), mt.num_layers, num_tokens, max_prompt_len)
        attention_weights = torch.zeros(len(data), mt.num_layers, mt.model.config.num_attention_heads, num_tokens, max_prompt_len)

    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                         do_sample=args.do_sample,
                                         temperature=args.temperature,
                                         stop_strings=args.stop_strings,
                                         pad_token_id=mt.tokenizer.pad_token_id)

    new_data = []
    for index, instance in tqdm(enumerate(data)):
        # 1. tokenize
        if args.template == 'plain':
            assert 'prompt' in instance
            prompt = instance['prompt']
        elif args.template == 'chat':
            assert 'messages' in instance
            prompt = mt.tokenizer.apply_chat_template(instance['messages'],
                                                      add_generation_prompt=True,
                                                      tokenize=False)

        elif args.template == 'continuation':
            assert 'messages' in instance
            prompt = mt.tokenizer.apply_chat_template(instance['messages'],
                                                      continue_final_message=True,
                                                      tokenize=False)
            if prompt.endswith('<|im_end|>\n'):
                prompt = prompt[:-len('<|im_end|>\n')]
        else:
            raise NotImplementedError

        if args.fixed_token is not None:
            input_len = len(mt.tokenizer.encode(prompt, add_special_tokens=False))
            prompt += args.fixed_token

        inputs = mt.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(mt.device)
        q_len = inputs['input_ids'].shape[1]

        if args.fixed_token is None:
            input_len = q_len

        # 2. generate
        if args.do_generate:
            output = mt.model.generate(**inputs,
                                       tokenizer=mt.tokenizer,
                                       generation_config=generation_config,
                                       use_cache=True,  # ✅ 加速推理
                                       logits_processor=LogitsProcessorList()  # ✅ 确保 logits 不被修改
                                       )
            decoded_output = mt.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
            instance['prompt'] = prompt
            instance['response'] = decoded_output
            if args.fixed_token is not None:
                instance['response'] = args.fixed_token + instance['response']
            instance['activations_index'] = index
            new_data.append(instance)

        if args.save_activations:
            activations = get_attentions(mt, inputs, position=input_len - 1, module=args.module)
            if args.module == 'attn':
                assert activations['attentions'].shape[2] == q_len
                attentions[index, :, :, :q_len] = activations['attentions']
            assert activations['attention_weights'].shape[3] == q_len
            attention_weights[index, :, :, :, :q_len] = activations['attention_weights']

        if index == 0:
            print(f'Prompt: {prompt}')
            print(f'Input ids: {inputs["input_ids"]}')
            if args.do_generate:
                print(f'Generated output: {instance["response"]}')
            if args.save_activations:
                if args.module == 'attn':
                    print(f'Attention shape: {activations["attentions"].shape}')
                print(f'Attention weights shape: {activations["attention_weights"].shape}')

    # save
    if not os.path.exists(os.path.dirname(args.output_filename)):
        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    if args.save_activations:
        if args.module == 'attn':
            save_path = os.path.join(os.path.dirname(args.output_filename), f'attentions.h5')
            print(attentions.shape)
            attentions_array = attentions.cpu().numpy()
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('activations', data=attentions_array)
            print(f'Saved attentions to {save_path}')

        attn_weights_save_path = os.path.join(os.path.dirname(args.output_filename), f'attn_weights.h5')
        print(attention_weights.shape)
        attention_weights_array = attention_weights.cpu().numpy()
        with h5py.File(attn_weights_save_path, 'w') as f:
            f.create_dataset('attn_weights', data=attention_weights_array)
        print(f'Saved attention weights to {attn_weights_save_path}')

    if args.do_generate:
        with open(args.output_filename, 'w') as f:
            for instance in new_data:
                f.write(json.dumps(instance) + '\n')
        print(f'Saved output to {args.output_filename}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--input_filename', type=str, default='data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl')
    parser.add_argument('--output_filename', type=str, default='temp')
    parser.add_argument('--template', type=str, choices=['plain', 'chat', 'continuation'], default='continuation')
    parser.add_argument('--fixed_token', type=str, default=None)

    # generation
    parser.add_argument('--do_generate', action='store_true')
    # parser.add_argument('--get_activations_while_generate', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--stop_strings', type=str, nargs='+', default=["<|eot_id|>", "<|im_end|>", "<|endoftext|>"])

    # activations
    parser.add_argument('--save_activations', type=bool, default=True)
    parser.add_argument('--position', type=int, default=-1)
    parser.add_argument('--module', type=str, choices=['attn', 'attn_weights'], default='attn_weights')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    args.stop_strings.append('\n')
    assert not args.do_generate
    print(args)
    return args


if __name__ == '__main__':
    main()
