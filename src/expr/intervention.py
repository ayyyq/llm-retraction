import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(src_path)
sys.path.append(src_path)

import json
import h5py
import tqdm
import argparse
import torch
from transformers import GenerationConfig, LogitsProcessorList
import random
random.seed(42)

from utils.general_utils import *
from utils.patchscopes_utils import *


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


def main(args):
    mt = load(args.model_path, args.attn_implementation, dtype=args.dtype)

    # 1. 计算mean shift方向
    data = [json.loads(line) for line in open(args.train_filename)]
    if args.wo_end:
        correct_indices = [instance['activations_index'] for instance in data if instance['label'] and not instance['end_tag']]
        incorrect_indices = [instance['activations_index'] for instance in data if not instance['label'] and not instance['end_tag']]
    else:
        correct_indices = [instance['activations_index'] for instance in data if instance['label']]
        incorrect_indices = [instance['activations_index'] for instance in data if not instance['label']]
    print(len(correct_indices))
    print(len(incorrect_indices))
    if args.train_num > 0:
        correct_indices = random.sample(correct_indices, args.train_num)
        incorrect_indices = random.sample(incorrect_indices, args.train_num)
        print('After sampling:', len(correct_indices), len(incorrect_indices))

    activations_path = os.path.join(os.path.dirname(args.train_filename), args.activations_name)
    with h5py.File(activations_path, 'r') as f:
        activations = torch.tensor(f['activations'][:], dtype=mt.model.dtype)

    pos_acts = activations[correct_indices]  # [bsz, n_layers, hidden_size]
    neg_acts = activations[incorrect_indices]  # [bsz, n_layers, hidden_size]

    pos_mean, neg_mean = pos_acts.mean(dim=0), neg_acts.mean(dim=0)
    v = pos_mean - neg_mean  # [n_layers, hidden_size]
    if args.ablation:
        v = v / v.norm(dim=1, keepdim=True)
    print('v shape:', v.shape)

    if args.direction_iid:
        raise NotImplementedError
        direction_path = os.path.join(os.path.dirname(args.train_filename), 'hs_iid_directions.h5')

        if os.path.exists(direction_path):
            with h5py.File(direction_path, 'r') as f:
                v = torch.tensor(f['directions'][:])
            print('Load iid direction from', direction_path)
        else:
            v_corrected = []
            for layer in range(mt.num_layers):
                centered_data_layer = torch.cat([pos_acts[:, layer, :] - pos_mean[layer],
                                                 neg_acts[:, layer, :] - neg_mean[layer]], dim=0)  # [bsz, hidden_size]

                covariance_layer = (centered_data_layer.T @ centered_data_layer) / centered_data_layer.shape[0]  # [hidden_size, hidden_size]
                inv_layer = torch.linalg.pinv(
                    covariance_layer + torch.eye(covariance_layer.shape[0], device=covariance_layer.device) * 1e-3)

                v_corrected.append(inv_layer @ v[layer])  # 计算修正后的 v[layer]
                print('Layer', layer, 'v_corrected shape:', v_corrected[-1].shape)

            v_corrected = torch.stack(v_corrected)  # [n_layers, hidden_size]
            v = v_corrected  # 更新 v
            print('Corrected v shape:', v.shape)

            direction_array = v.cpu().numpy()
            with h5py.File(direction_path, 'w') as f:
                f.create_dataset('directions', data=direction_array)
            print('Save iid direction to', direction_path)

    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                         do_sample=args.do_sample,
                                         temperature=args.temperature,
                                         stop_strings=args.stop_strings,
                                         pad_token_id=mt.tokenizer.pad_token_id)

    # 定义hook函数
    def intervene_hs(target_layer, target_position=-1):
        def intervention_hook(module, input, output):
            output_len = len(output[0][0])

            if output_len == 1 and not args.multitoken_steer:
                return

            alpha = args.alpha

            # print('Before intervention', output[0][0, -1])
            if args.ablation:
                v_layer = v[target_layer].to(device=output[0].device, dtype=output[0].dtype)
                output[0][0, target_position] -= (output[0][0, target_position] @ v_layer) * v_layer
            elif args.positive_steer:
                output[0][0, target_position] += alpha * v[target_layer].to(output[0].device)
            else:
                output[0][0, target_position] -= alpha * v[target_layer].to(output[0].device)  # 直接修改 output
            # print('After intervention', output[0][0, -1])
        return intervention_hook

    # inference
    test_data = [json.loads(line) for line in open(args.test_filename)]
    if args.debug:
        test_data = test_data[:10]

    new_data = []
    if args.save_activations:
        activations = torch.zeros(len(test_data), mt.num_layers, mt.model.config.hidden_size)

    if isinstance(args.target_layer, list):
        target_layers = args.target_layer
    else:
        if args.multilayer_steer == 1:
            target_layers = list(range(args.target_layer, mt.num_layers - 1))  # TODO: 暂时不考虑最后一层
        elif args.multilayer_steer == -1:
            target_layers = list(range(0, args.target_layer + 1))
        else:
            target_layers = [args.target_layer]
    print('Target layers:', target_layers)

    for index, instance in tqdm.tqdm(enumerate(test_data)):

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
        if args.fixed_token is None:
            input_len = inputs['input_ids'].shape[1]

        hooks = []
        for target_layer in target_layers:
            hook = intervene_hs(target_layer, target_position=input_len - 1)
            hooks.append(mt.model.model.layers[target_layer].register_forward_hook(hook))

        if args.do_generate:
            if args.ban_first_token:
                banned_token = '\'s'
                banned_token_id = mt.tokenizer.encode(banned_token, add_special_tokens=False)
                if index == 0:
                    print('banned_token:', banned_token, banned_token_id)
                banned_token_id = banned_token_id[0]

                logits_processor = LogitsProcessorList([
                    FirstTokenBanProcessor(banned_token_ids=[banned_token_id], initial_length=inputs['input_ids'].shape[1])
                ])
            else:
                logits_processor = None
            output = mt.model.generate(**inputs,
                                       tokenizer=mt.tokenizer,
                                       generation_config=generation_config,
                                       logits_processor=logits_processor)
            decoded_output = mt.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
            instance['prompt'] = prompt
            instance['response'] = decoded_output
            new_data.append(instance)

        if args.save_activations:
            output = mt.model(**inputs, output_hidden_states=True)
            last_token_activations = [
                output["hidden_states"][layer + 1][0, -1] for layer in range(mt.num_layers)
            ]
            last_token_activations = torch.stack(last_token_activations)  # [n_layers, hidden_size]
            activations[index] = last_token_activations

        if index == 0:
            if args.ablation:
                print(f"Ablation: layer {target_layers}, position {input_len - 1}")
            elif args.positive_steer:
                print(f'Positive steer: layer {target_layers}, position {input_len - 1}')
            else:
                print(f'Negative steer: layer {target_layers}, position {input_len - 1}')
            print(f'Prompt: {prompt}')
            print(f'Input ids: {inputs["input_ids"]}')
            if args.do_generate:
                print(f'Generated output: {decoded_output}')
            if args.save_activations:
                print(f'Activations shape: {last_token_activations.shape}')
        elif args.debug:
            print(prompt)
            print(decoded_output)

        remove_hooks(hooks)

    if isinstance(args.target_layer, list):
        output_filename = os.path.join(args.save_dir, f'layer{args.target_layer[0]}-{args.target_layer[-1]}_alpha{args.alpha}', 'output.jsonl')
    else:
        output_filename = os.path.join(args.save_dir, f'layer{args.target_layer}_alpha{args.alpha}', 'output.jsonl')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if args.do_generate:
        with open(output_filename, 'w') as f:
            for instance in new_data:
                f.write(json.dumps(instance) + '\n')
        print('Save results to', output_filename)

    if args.save_activations:
        save_path = os.path.join(os.path.dirname(output_filename), f'hs_activations.h5')
        print(activations.shape)
        activations_array = activations.cpu().numpy()
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('activations', data=activations_array)
        print(f'Saved activations to {save_path}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--model_path', type=str, default='/mnt/nfs1/yuqing/meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')

    parser.add_argument("--train_filename", type=str, default='/home/yuqing/project/LLMDecomp/probe-outputs/universal_truthfulness/truthfulness_train/Llama-3.1-8B-Instruct/t0/output.jsonl')
    parser.add_argument("--test_filename", type=str, default='/home/yuqing/project/LLMDecomp/data_collection/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl')
    parser.add_argument('--template', type=str, choices=['plain', 'chat', 'continuation'], default='continuation')
    parser.add_argument("--activations_name", type=str, default='hs_activations.h5')
    parser.add_argument("--save_dir", type=str, default='temp')  # currently, 默认negative steer + multilayer_steer == 0，multilayer_steer == 1 -> multilayer, multilayer_steer == -1 -> multilayer-1
    parser.add_argument("--train_num", type=int, default=-1)

    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--stop_strings', type=str, nargs='+', default=['<|eot_id|>', '<|im_end|>', "<|endoftext|>"])
    parser.add_argument('--fixed_token', type=str, default=None)
    parser.add_argument('--ban_first_token', action='store_true')

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--target_layer', type=int, nargs='+', default=[-1])
    parser.add_argument('--positive_steer', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--multilayer_steer', type=int, choices=[-1, 0, 1], default=0)
    parser.add_argument('--multitoken_steer', action='store_true')
    parser.add_argument('--direction_iid', action='store_true')
    parser.add_argument('--wo_end', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_do_generate', action='store_true')
    parser.add_argument('--save_activations', action='store_true')

    args = parser.parse_args()
    args.do_generate = not args.no_do_generate

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

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.target_layer[0] == -1:
        for target_layer in [2, 6, 10, 14, 18, 22, 26, 30]:
            args.target_layer = target_layer
            print(args)
            main(args)
    else:
        if len(args.target_layer) == 1:
            args.target_layer = args.target_layer[0]
        print(args)
        main(args)
