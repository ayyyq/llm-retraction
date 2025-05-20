import os
import sys
from typing import Optional, Tuple

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(src_path)
sys.path.append(src_path)

import json
import h5py
import tqdm
import argparse

import math
import torch
import torch.nn.functional as F
from torch import nn
import transformers.models.llama.modeling_llama as modeling_llama
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2
import transformers.models.olmo2.modeling_olmo2 as modeling_olmo2
from transformers import Cache, GenerationConfig

from utils.general_utils import *
from utils.patchscopes_utils import *


class MyLlamaAttention(modeling_llama.LlamaAttention):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        if hidden_states.shape[1] != 1:
            if hasattr(self, "value_patch") and self.value_patch is not None:
                value_states[0, self.value_position, :] = self.value_patch
                self.value_patch = None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            modeling_llama.logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        if hidden_states.shape[1] != 1:
            if hasattr(self, "attn_patch") and self.attn_patch is not None:
                if hasattr(self, "topk_heads") and self.topk_heads is not None:
                    # attn_patch: [num_heads, num_tokens, seq_len]
                    for i, heads in self.topk_heads.items():
                        heads = torch.tensor(heads)
                        position_i = q_len - self.attn_patch.shape[1] + i
                        attn_weights[0, heads, position_i, :] = self.attn_patch[heads, i, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                    self.topk_heads = None
                elif hasattr(self, "heads_positions") and self.heads_positions is not None:
                    for i in self.heads_positions:
                        position_i = q_len - self.attn_patch.shape[1] + i
                        attn_weights[0, :, position_i, :] = self.attn_patch[:, i, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                    self.heads_positions = None
                else:
                    raise NotImplementedError
                    attn_weights[0, :, self.token_positions, :] = self.attn_patch[:, self.token_positions, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                self.attn_patch = None

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MyQwen2Attention(modeling_qwen2.Qwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if hidden_states.shape[1] != 1:
            if hasattr(self, "value_patch") and self.value_patch is not None:
                value_states[0, self.value_position, :] = self.value_patch
                self.value_patch = None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.layer_idx == self.config.num_hidden_layers - 1 and self.num_heads == 28 and self.hidden_size == 3584:
            # for Qwen2-7B models and the last layer only
            attn_weights = torch.matmul(query_states / math.sqrt(self.head_dim), key_states.transpose(2, 3))
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        if hidden_states.shape[1] != 1:
            if hasattr(self, "attn_patch") and self.attn_patch is not None:
                if hasattr(self, "topk_heads") and self.topk_heads is not None:
                    # attn_patch: [num_heads, num_tokens, seq_len]
                    for i, heads in self.topk_heads.items():
                        heads = torch.tensor(heads)
                        position_i = q_len - self.attn_patch.shape[1] + i
                        attn_weights[0, heads, position_i, :] = self.attn_patch[heads, i, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                    self.topk_heads = None
                elif hasattr(self, "heads_positions") and self.heads_positions is not None:
                    for i in self.heads_positions:
                        position_i = q_len - self.attn_patch.shape[1] + i
                        attn_weights[0, :, position_i, :] = self.attn_patch[:, i, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                    self.heads_positions = None
                else:
                    raise NotImplementedError
                self.attn_patch = None

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MyOlmo2Attention(modeling_olmo2.Olmo2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if hidden_states.shape[1] != 1:
            if hasattr(self, "value_patch") and self.value_patch is not None:
                value_states[0, self.value_position, :] = self.value_patch
                self.value_patch = None

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_olmo2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)

        if hidden_states.shape[1] != 1:
            q_len = hidden_states.shape[1]
            if hasattr(self, "attn_patch") and self.attn_patch is not None:
                if hasattr(self, "topk_heads") and self.topk_heads is not None:
                    # attn_patch: [num_heads, num_tokens, seq_len]
                    for i, heads in self.topk_heads.items():
                        heads = torch.tensor(heads)
                        position_i = q_len - self.attn_patch.shape[1] + i
                        attn_weights[0, heads, position_i, :] = self.attn_patch[heads, i, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                    self.topk_heads = None
                elif hasattr(self, "heads_positions") and self.heads_positions is not None:
                    for i in self.heads_positions:
                        position_i = q_len - self.attn_patch.shape[1] + i
                        attn_weights[0, :, position_i, :] = self.attn_patch[:, i, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                    self.heads_positions = None
                else:
                    raise NotImplementedError
                    attn_weights[0, :, self.token_positions, :] = self.attn_patch[:, self.token_positions, :q_len].to(dtype=attn_weights.dtype, device=attn_weights.device)
                self.attn_patch = None

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def load(model_path, model=None, attn_implementation='flash_attention_2'):
    sos_tok = False
    torch_dtype = torch.float16

    mt = ModelAndTokenizer(
        model_name=model_path,
        model=model,
        low_cpu_mem_usage=False,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation
    )
    mt.model.eval()
    torch.set_grad_enabled(False)
    return mt


def main(args):
    assert args.attn_implementation == 'eager'

    if 'llama' in args.model_path.lower():
        modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = MyLlamaAttention
    elif 'qwen' in args.model_path.lower():
        modeling_qwen2.QWEN2_ATTENTION_CLASSES["eager"] = MyQwen2Attention
    elif 'olmo' in args.model_path.lower():
        modeling_olmo2.Olmo2Attention = MyOlmo2Attention
    mt = load(args.model_path, model=None, attn_implementation=args.attn_implementation)
    print(mt.model.model.layers[0].self_attn)

    generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                         do_sample=args.do_sample,
                                         temperature=args.temperature,
                                         stop_strings=args.stop_strings,
                                         pad_token_id=mt.tokenizer.pad_token_id)

    # inference
    test_data = [json.loads(line) for line in open(args.test_filename)]
    print(f'Test data size: {len(test_data)}')
    if args.debug:
        test_data = test_data[:10]

    if args.patch_attn_weights:
        activations_path = os.path.join(os.path.dirname(args.train_filename), 'attn_weights.h5')
        with h5py.File(activations_path, 'r') as f:
            attn_weights = torch.tensor(f['attn_weights'][:])  # bsz, num_layers, num_heads, num_tokens, seq_len
        print(f'Loaded attention weights from {activations_path}, shape: {attn_weights.shape}')
        assert attn_weights.shape[0] == len(test_data), f'Attention weights shape: {attn_weights.shape}, test data size: {len(test_data)}'

    if args.patch_value_vectors:
        activations_path = os.path.join(os.path.dirname(args.train_filename), 'attn_value_vectors.h5')
        with h5py.File(activations_path, 'r') as f:
            value_vectors = torch.tensor(f['attn_value_vectors'][:])  # [bsz, num_layers, value_head_dim]
        print(f'Loaded value vector from {activations_path}, shape: {value_vectors.shape}')
        assert value_vectors.shape[0] == len(test_data), f'Value vectors shape: {value_vectors.shape}, test data size: {len(test_data)}'

    topk_heads_dict = {}
    if args.patch_attn_weights:
        if args.heads_topk != -1:
            for position, path in args.topk_heads_path.items():
                topk_heads = [json.loads(line) for line in open(path)][:args.heads_topk]
                for h in topk_heads:
                    layer = int(h['layer'])
                    head = int(h['head'])

                    if layer not in topk_heads_dict:
                        topk_heads_dict[layer] = {}

                    if position not in topk_heads_dict[layer]:
                        topk_heads_dict[layer][position] = []
                    topk_heads_dict[layer][position].append(head)
                print(f'Loaded top {args.heads_topk} heads from {path}')

        if args.heads_topk != -1:
            first_name = f'patch_attn_top{args.heads_topk}_heads'
        else:
            first_name = 'patch_attn'
        mid_name = 'pos' + '-'.join([str(i) for i in args.heads_positions])

        if args.patch_value_vectors:
            first_name += '_and_value_vectors'
            if args.value_vectors_layers[0] != -1:
                mid_name += f'_and_layer{args.value_vectors_layers[0]}-{args.value_vectors_layers[-1]}'

        first_name = os.path.join(first_name, mid_name)

    if not args.patch_attn_weights and args.patch_value_vectors:
        first_name = 'patch_value_vectors'
        if args.value_vectors_layers[0] != -1:
            mid_name = f'layer{args.value_vectors_layers[0]}-{args.value_vectors_layers[-1]}'
            first_name = os.path.join(first_name, mid_name)

    if args.fixed_token is not None:
        assert 'fixed' in args.train_filename
        output_filename = os.path.join(os.path.dirname(args.train_filename), first_name, 'output.jsonl')
    else:
        output_filename = os.path.join(os.path.dirname(args.train_filename), 'wo_fixed_token', first_name, 'output.jsonl')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    print(f'Will be saved to {output_filename}')

    new_data = []

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

        for i, layer in enumerate(mt.model.model.layers):
            if args.patch_attn_weights:
                if topk_heads_dict:
                    if i in topk_heads_dict:
                        layer.self_attn.topk_heads = topk_heads_dict[i]
                        layer.self_attn.attn_patch = attn_weights[index][i]  # shape: [num_heads, num_tokens, seq_len]
                else:
                    layer.self_attn.heads_positions = args.heads_positions
                    layer.self_attn.attn_patch = attn_weights[index][i]  # shape: [num_heads, num_tokens, seq_len]

            if args.patch_value_vectors:
                if args.value_vectors_layers[0] != -1 and i not in args.value_vectors_layers:
                    continue
                layer.self_attn.value_position = input_len - 1
                layer.self_attn.value_patch = value_vectors[index][i]  # shape: [value_head_dim]

        if index == 0:
            if args.patch_attn_weights:
                if topk_heads_dict:
                    print('Patching attention weights at topk heads', topk_heads_dict)
                else:
                    print('Patching attention weights at positions', args.heads_positions)
            if args.patch_value_vectors:
                if args.value_vectors_layers[0] != -1:
                    print('Patching value vectors at layers', args.value_vectors_layers, 'and position', input_len - 1)
                else:
                    print('Patching value vectors at all layers at position', input_len - 1)

        output = mt.model.generate(**inputs,
                                   tokenizer=mt.tokenizer,
                                   generation_config=generation_config)
        decoded_output = mt.tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
        instance['prompt'] = prompt
        instance['response'] = decoded_output
        new_data.append(instance)

        if index == 0:
            print(f'Prompt: {prompt}')
            print(f'Input ids: {inputs["input_ids"]}')
            print(f'Generated output: {decoded_output}')

    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save results to', output_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='allenai/OLMo-2-1124-7B-Instruct')
    parser.add_argument('--attn_implementation', type=str, default='eager')

    parser.add_argument("--train_filename", type=str, default='intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/negative_steer/wikidata_test_continuation/layer8-30_alpha1.5/output.jsonl')
    parser.add_argument("--test_filename", type=str, default='data/wikidata/wikidata_continuation/OLMo-2-1124-7B-Instruct/wikidata_test_continuation.jsonl')
    parser.add_argument('--template', type=str, choices=['plain', 'chat', 'continuation'], default='continuation')

    parser.add_argument("--patch_attn_weights", action='store_true')
    parser.add_argument("--topk_heads_dir", type=str, default='top_heads')
    parser.add_argument("--heads_positions", type=int, nargs='+', default=[0, 3])
    parser.add_argument("--heads_topk", type=int, default=-1)
    parser.add_argument("--patch_value_vectors", action='store_true')
    parser.add_argument("--value_vectors_layers", type=int, nargs='+', default=[-1])

    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--stop_strings', type=str, nargs='+', default=["<|eot_id|>", "<|im_end|>", "<|endoftext|>"])
    parser.add_argument('--fixed_token', type=str, default=None)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    args.stop_strings.append('\n')
    return args


if __name__ == '__main__':
    args = parse_args()

    topk_heads_paths = {
        0: f'{args.topk_heads_dir}/top100_heads_pos0.jsonl',
        1: f'{args.topk_heads_dir}/top100_heads_pos1.jsonl'
    }
    args.topk_heads_path = None
    if args.patch_attn_weights and args.heads_topk != -1:
        args.topk_heads_path = {i: topk_heads_paths[i] for i in args.heads_positions}

    print(args)

    main(args)
