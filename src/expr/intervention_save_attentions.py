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
from transformers import StoppingCriteria, GenerationConfig

from utils.general_utils import *
from utils.patchscopes_utils import *


def load(model_path, attn_implementation='flash_attention_2'):
    sos_tok = False
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
    assert args.attn_implementation == 'eager'
    mt = load(args.model_path, args.attn_implementation)

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

    activations_path = os.path.join(os.path.dirname(args.train_filename), args.activations_name)
    with h5py.File(activations_path, 'r') as f:
        activations = torch.tensor(f['activations'][:])

    pos_acts = activations[correct_indices]  # [bsz, n_layers, hidden_size]
    neg_acts = activations[incorrect_indices]  # [bsz, n_layers, hidden_size]

    pos_mean, neg_mean = pos_acts.mean(dim=0), neg_acts.mean(dim=0)
    v = pos_mean - neg_mean  # [n_layers, hidden_size]
    print('v shape:', v.shape)

    # 定义hook函数
    def intervene_hs(target_layer, target_position=-1):
        def intervention_hook(module, input, output):
            output_len = len(output[0][0])

            if output_len == 1 and not args.multitoken_steer:
                return

            alpha = args.alpha

            # print('Before intervention', output[0][0, -1])
            if args.positive_steer:
                output[0][0, target_position] += alpha * v[target_layer].to(output[0].device)
            else:
                output[0][0, target_position] -= alpha * v[target_layer].to(output[0].device)  # 直接修改 output
            # print('After intervention', output[0][0, -1])
        return intervention_hook

    # inference
    test_data = [json.loads(line) for line in open(args.test_filename)]
    print(f'Test data size: {len(test_data)}')
    if args.debug:
        test_data = test_data[:10]

    if args.template == 'plain':
        max_prompt_len = max(
            [len(mt.tokenizer.encode(instance['prompt'], add_special_tokens=False)) for instance in test_data])
    elif args.template == 'chat':
        max_prompt_len = max(
            [len(mt.tokenizer.apply_chat_template(instance['messages'], add_generation_prompt=True, tokenize=True))
             for instance in test_data])
    elif args.template == 'continuation':
        max_prompt_len = max(
            [len(mt.tokenizer.apply_chat_template(instance['messages'], continue_final_message=True, tokenize=True))
             for instance in test_data])
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
    attentions = torch.zeros(len(test_data), mt.num_layers, num_tokens, max_prompt_len)
    attention_weights = torch.zeros(len(test_data), mt.num_layers, mt.model.config.num_attention_heads, num_tokens, max_prompt_len)
    head_dim = getattr(mt.model.config, "head_dim", mt.model.config.hidden_size // mt.model.config.num_attention_heads)
    value_vectors = torch.zeros(len(test_data), mt.num_layers, mt.model.config.num_key_value_heads * head_dim)

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
        elif args.fixed_token == ' was born in':
            assert inputs['input_ids'].shape[1] == input_len + 3
        elif args.fixed_token == ' is':
            assert inputs['input_ids'].shape[1] == input_len + 1

        hooks = []
        _attentions = []
        _value_vectors = []
        def wrap_forward_with_cache_input(attn_module):
            if getattr(attn_module, '_is_wrapped', False):
                return

            attn_module._original_forward = attn_module.forward

            @wraps(attn_module._original_forward)
            def new_forward(*args, **kwargs):
                # 支持从 args 或 kwargs 拿 input
                if args:
                    hidden_states = args[0]
                else:
                    hidden_states = kwargs["hidden_states"]

                attn_module._cached_hidden_states = hidden_states
                return attn_module._original_forward(*args, **kwargs)

            attn_module.forward = new_forward
            attn_module._is_wrapped = True

        def store_attn(start_position, olmo=False):
            def store_attn_hook(module, input, output):
                # query token i (position), key/value token j
                hidden_states = module._cached_hidden_states
                bsz, q_len, hidden_dim = hidden_states.size()
                assert bsz == 1, "This hook only supports batch size = 1"
                num_tokens = q_len - start_position

                num_key_value_heads = module.num_key_value_heads
                head_dim = module.head_dim

                value_states = module.v_proj(hidden_states)
                _value_vectors.append(value_states[0, start_position, :].clone().detach())  # [value_head_dim]

                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
                value_states = repeat_kv(value_states, module.num_key_value_groups)

                v_j = value_states[0]  # [num_heads, seq_len, head_dim]  # k_len == v_len

                attn_weights = output[1]
                # a_ij = attn_weights[0, :, -1, :]  # [num_heads, seq_len]
                a_ij = attn_weights[0, :, start_position:, :]  # [num_heads, num_tokens, seq_len]

                # a_i = a_ij[:, :, None] * v_j  # [num_heads, seq_len, head_dim]
                a_i = a_ij[:, :, :, None] * v_j[:, None, :, :]  # [num_heads, num_tokens, seq_len, head_dim]

                # a_i = a_i.permute(1, 0, 2).contiguous()  # [seq_len, num_heads, head_dim]
                a_i = a_i.permute(1, 2, 0, 3).contiguous()  # [num_tokens, seq_len, num_heads, head_dim]
                a_i = a_i.reshape(num_tokens, q_len, -1)  # [num_tokens, seq_len, hidden_dim]

                a_i = module.o_proj(a_i)
                a_i_norm = a_i.norm(dim=-1)  # [num_tokens, seq_len]
                _attentions.append(a_i_norm.clone().detach())

            def store_olmo_attn_hook(module, input, output):
                # query token i (position), key/value token j
                hidden_states = module._cached_hidden_states

                value_states = module.v_proj(hidden_states)
                assert len(value_states.shape) == 3
                _value_vectors.append(value_states[0, start_position, :].clone().detach())  # [value_head_dim]

            if olmo:
                return store_olmo_attn_hook
            else:
                return store_attn_hook

        for target_layer in target_layers:
            hook = intervene_hs(target_layer, target_position=input_len - 1)
            hooks.append(mt.model.model.layers[target_layer].register_forward_hook(hook))

        for layer in mt.model.model.layers:
            wrap_forward_with_cache_input(layer.self_attn)
            hooks.append(layer.self_attn.register_forward_hook(store_attn(input_len - 1, olmo=args.olmo)))

        output = mt.model(**inputs, output_attentions=True)
        _attention_weights = [
            output["attentions"][layer][0, :, input_len - 1:] for layer in range(mt.num_layers)
        ]

        # _attentions = torch.stack(_attentions, dim=0)  # [num_layers, num_tokens, seq_len]
        _attention_weights = torch.stack(_attention_weights, dim=0)  # [num_layers, num_heads, num_tokens, seq_len]
        _value_vectors = torch.stack(_value_vectors, dim=0)  # [num_layers, value_head_dim]

        if index == 0:
            if args.positive_steer:
                print(f'Positive steer: layer {target_layers}, position {input_len - 1}')
            else:
                print(f'Negative steer: layer {target_layers}, position {input_len - 1}')
            print(f'Prompt: {prompt}')
            print(f'Input ids: {inputs["input_ids"]}')
            # print(f'Attention shape: {_attentions.shape}')
            print(f'Attention weights shape: {_attention_weights.shape}')
            print(f'Value vectors shape: {_value_vectors.shape}')

        remove_hooks(hooks)
        unwrap_attn_modules(mt)

        # assert _attentions.shape[1] == input_len
        # q_len = _attentions.shape[2]
        # assert _attention_weights.shape[3] == q_len
        # attentions[index, :, :, :q_len] = _attentions
        # assert _attention_weights.shape[2] == input_len
        q_len = _attention_weights.shape[3]
        attention_weights[index, :, :, :, :q_len] = _attention_weights

        assert value_vectors.shape[-1] == _value_vectors.shape[-1]
        value_vectors[index] = _value_vectors

    if isinstance(args.target_layer, list):
        output_filename = os.path.join(args.save_dir, f'layer{args.target_layer[0]}-{args.target_layer[-1]}_alpha{args.alpha}', 'output.jsonl')
    else:
        output_filename = os.path.join(args.save_dir, f'layer{args.target_layer}_alpha{args.alpha}', 'output.jsonl')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # save_path = os.path.join(os.path.dirname(output_filename), f'attentions.h5')
    # print(attentions.shape)
    # attentions_array = attentions.cpu().numpy()
    # with h5py.File(save_path, 'w') as f:
    #     f.create_dataset('activations', data=attentions_array)
    # print(f'Saved attentions to {save_path}')

    attn_weights_save_path = os.path.join(os.path.dirname(output_filename), f'attn_weights.h5')
    print(attention_weights.shape)
    attention_weights_array = attention_weights.cpu().numpy()
    with h5py.File(attn_weights_save_path, 'w') as f:
        f.create_dataset('attn_weights', data=attention_weights_array)
    print(f'Saved attention weights to {attn_weights_save_path}')

    attn_value_vectors_save_path = os.path.join(os.path.dirname(output_filename), f'attn_value_vectors.h5')
    print(value_vectors.shape)
    value_vectors_array = value_vectors.cpu().numpy()
    with h5py.File(attn_value_vectors_save_path, 'w') as f:
        f.create_dataset('attn_value_vectors', data=value_vectors_array)
    print(f'Saved attention value vectors to {attn_value_vectors_save_path}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='/mnt/nfs1/yuqing/meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--attn_implementation', type=str, default='eager')

    parser.add_argument("--train_filename", type=str, default='/home/yuqing/project/LLMDecomp/probe-outputs/universal_truthfulness/truthfulness_train/Llama-3.1-8B-Instruct/t0/output.jsonl')
    parser.add_argument("--test_filename", type=str, default='/home/yuqing/project/LLMDecomp/data_collection/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl')
    parser.add_argument('--template', type=str, choices=['plain', 'chat', 'continuation'], default='continuation')
    parser.add_argument("--activations_name", type=str, default='hs_activations.h5')
    parser.add_argument("--save_dir", type=str, default='temp')  # currently, 默认negative steer + multilayer_steer == 0，multilayer_steer == 1 -> multilayer, multilayer_steer == -1 -> multilayer-1

    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--stop_strings', type=str, nargs='+', default=["<|eot_id|>", "<|im_end|>", "<|endoftext|>"])
    parser.add_argument('--fixed_token', type=str, default=None)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--target_layer', type=int, nargs='+', default=[-1])
    parser.add_argument('--positive_steer', action='store_true')
    parser.add_argument('--multilayer_steer', type=int, choices=[-1, 0, 1], default=0)
    parser.add_argument('--multitoken_steer', action='store_true')
    parser.add_argument('--direction_iid', action='store_true')
    parser.add_argument('--wo_end', action='store_true')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    args.stop_strings.append('\n')
    args.olmo = 'olmo' in args.model_path.lower()
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
