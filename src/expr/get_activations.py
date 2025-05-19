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


def load(model_path, attn_implementation='flash_attention_2', dtype='fp16'):
    sos_tok = False
    if dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    mt = ModelAndTokenizer(
        model_path,
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation
    )
    mt.model.eval()
    torch.set_grad_enabled(False)
    return mt


def main():
    args = parse_args()
    mt = load(args.model_path, args.attn_implementation, dtype=args.dtype)

    # read data
    data = [json.loads(line) for line in open(args.input_filename)]
    print(f'Loaded {len(data)} instances')
    if args.debug:
        # data = data[:10]
        data = [instance for instance in data if not instance['result']][:50]

    if args.get_activations_while_generate:
        activations = []
    else:
        activations = torch.zeros(len(data), mt.num_layers, mt.model.config.hidden_size)

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
            messages = instance['messages']
            if args.fixed_token is not None:
                messages[-1]['content'] += args.fixed_token

            prompt = mt.tokenizer.apply_chat_template(messages,
                                                      continue_final_message=True,
                                                      tokenize=False)
            if prompt.endswith('<|im_end|>\n'):
                prompt = prompt[:-len('<|im_end|>\n')]
        else:
            raise NotImplementedError

        inputs = mt.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(mt.device)
        input_len = inputs['input_ids'].shape[1]

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

        # 3. get last token activations
        elif args.get_activations_while_generate:
            activation, output = get_activations_while_generate(mt, inputs,
                                                                max_new_tokens=args.max_new_tokens,
                                                                module=args.module)
            activations.append(activation)
            response_tokens = mt.tokenizer.convert_ids_to_tokens(output[0, input_len:])
            assert len(response_tokens) == len(activation)
            decoded_output = mt.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
            instance['prompt'] = prompt
            instance['response'] = decoded_output
            instance['response_tokens'] = response_tokens
            instance['activations_index'] = index
            new_data.append(instance)

        if not args.get_activations_while_generate and args.save_activations:
            last_token_activations = get_activations(mt, inputs, position=args.position, module=args.module)
            activations[index] = last_token_activations

        if index == 0:
            print(f'Prompt: {prompt}')
            print(f'Input ids: {inputs["input_ids"]}')
            if args.do_generate or args.get_activations_while_generate:
                print(f'Generated output: {instance["response"]}')
            if not args.get_activations_while_generate and args.save_activations:
                print(f'Activations shape: {last_token_activations.shape}')

    # save
    if not os.path.exists(os.path.dirname(args.output_filename)):
        os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    if args.save_activations:
        save_path = os.path.join(os.path.dirname(args.output_filename), f'{args.module}_activations.h5')
        if args.get_activations_while_generate:
            with h5py.File(save_path, "w") as hf:
                dt = h5py.vlen_dtype(np.float32)
                dset = hf.create_dataset("activations", (len(activations),), dtype=dt)
                for i, activation in enumerate(activations):
                    assert activation.shape[1] == mt.num_layers
                    assert activation.shape[2] == mt.model.config.hidden_size
                    dset[i] = activation.cpu().to(torch.float32).numpy().flatten()
        else:
            print(activations.shape)
            activations_array = activations.cpu().numpy()
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('activations', data=activations_array)
        print(f'Saved activations to {save_path}')

    if args.do_generate or args.get_activations_while_generate:
        with open(args.output_filename, 'w') as f:
            for instance in new_data:
                f.write(json.dumps(instance) + '\n')
        print(f'Saved output to {args.output_filename}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
    parser.add_argument('--input_filename', type=str)
    parser.add_argument('--output_filename', type=str)
    parser.add_argument('--template', type=str, choices=['plain', 'chat', 'continuation'])
    parser.add_argument('--fixed_token', type=str, default=None)

    # generation
    parser.add_argument('--do_generate', action='store_true')
    parser.add_argument('--get_activations_while_generate', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--stop_strings', type=str, nargs='+', default=["<|eot_id|>", "<|im_end|>", "<|endoftext|>"])

    # activations
    parser.add_argument('--not_save_activations', action='store_true')
    parser.add_argument('--position', type=int, default=-1)
    parser.add_argument('--module', type=str, choices=['hs', 'mlp', 'attn', 'attn_head'], default='hs')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    args.save_activations = not args.not_save_activations

    args.stop_strings.append('\n')
    if args.fixed_token is not None:
        assert args.attn_implementation == 'eager'
    if 'Qwen' in args.model_path:
        if args.attn_implementation == 'flash_attention_2':
            args.dtype = 'bf16'
        elif args.attn_implementation == 'eager':
            args.dtype = 'fp16'
        else:
            raise NotImplementedError

    print(args)
    return args


if __name__ == '__main__':
    main()
