# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import tqdm
from utils.general_utils import decode_tokens
from utils.general_utils import make_inputs
from functools import wraps


# ##############
#
# Hooks
#
# ##############


def set_hs_patch_hooks_neox(
        model,
        hs_patch_config,
        module="hs",  # mlp, attn
        patch_input=False,
        skip_final_ln=False,
        generation_mode=False,
):
    """Neox patch hooks."""
    # when using mode.generate() the hidden states in the input are cached after
    # the first inference pass, and in the next steps the input/output are of
    # size 1. In these cases we don't need to patch anymore the previous hidden
    # states from the initial input, because they are cached, but we do need to
    # handle these cases in this call because this hook wraps the generation call.
    #
    # NOTE: To use generation mode, we must patch a position that is not the
    # first one. This is because in this case we don't know during generation if
    # we are handling the initial input or a future step and thus don't know if
    # a patching is needed or not.

    # if generation_mode:
    #     for i in hs_patch_config:
    #         for position_, _ in hs_patch_config[i]:
    #             assert position_ > 0

    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            # (batch, sequence, hidden_state)
            input_len = len(input[0][0])
            if generation_mode and input_len == 1:
                return
            for position_, hs_ in position_hs:
                input[0][0, position_] = hs_

        def post_hook(module, input, output):
            if "skip_ln" in name:
                # output: (batch, sequence, hidden_state)
                output_len = len(output[0])
            else:
                # output[0]: (batch, sequence, hidden_state)
                output_len = len(output[0][0])

            if generation_mode and output_len == 1:
                return
            for position_, hs_ in position_hs:
                if "skip_ln" in name:
                    output[0][position_] = hs_
                else:
                    output[0][0, position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for i in hs_patch_config:
        if patch_input:
            hooks.append(
                model.gpt_neox.layers[i].register_forward_pre_hook(
                    patch_hs(
                        f"patch_hs_{i}",
                        hs_patch_config[i],
                        patch_input,
                        generation_mode,
                    )
                )
            )
        else:
            # when patching a last-layer representation to the last layer of the
            # same model, the final layer norm is not needed because it was already
            # applied (assuming that the representation for patching was obtained by
            # setting output_hidden_representations to True).
            if skip_final_ln and i == len(model.gpt_neox.layers) - 1:
                hooks.append(
                    model.gpt_neox.final_layer_norm.register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}_skip_ln",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )
            else:
                hooks.append(
                    model.gpt_neox.layers[i].register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )

    return hooks


def set_hs_patch_hooks_llama(
        model,
        hs_patch_config,
        module="hs",  # mlp, attn
        patch_input=False,
        skip_final_ln=False,
        generation_mode=False,
):
    """Llama patch hooks."""

    # when using mode.generate() the hidden states in the input are cached after
    # the first inference pass, and in the next steps the input/output are of
    # size 1. In these cases we don't need to patch anymore the previous hidden
    # states from the initial input, because they are cached, but we do need to
    # handle these cases in this call because this hook wraps the generation call.
    #
    # NOTE: To use generation mode, we must patch a position that is not the
    # first one. This is because in this case we don't know during generation if
    # we are handling the initial input or a future step and thus don't know if
    # a patching is needed or not.

    # if generation_mode:
    #     for i in hs_patch_config:
    #         for position_, _ in hs_patch_config[i]:
    #             assert position_ > 0

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            # (batch, sequence, hidden_state)
            input_len = len(input[0][0])
            if generation_mode and input_len == 1:
                return
            for position_, hs_ in position_hs:
                input[0][0, position_] = hs_

        def post_hook(module, input, output):
            if "skip_ln" in name or "mlp" in name:
                # output: (batch, sequence, hidden_state)
                output_len = len(output[0])
            else:
                # output[0]: (batch, sequence, hidden_state)
                output_len = len(output[0][0])

            if generation_mode and output_len == 1:
                return
            for position_, hs_ in position_hs:
                if "skip_ln" in name or "mlp" in name:
                    output[0][position_] = hs_
                else:
                    output[0][0, position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for i in hs_patch_config:
        patch_hook = patch_hs(
            f"patch_{module}_{i}",
            position_hs=hs_patch_config[i],
            patch_input=patch_input,
            generation_mode=generation_mode,
        )
        if patch_input:
            if module == "hs":
                hooks.append(
                    model.model.layers[i].register_forward_pre_hook(patch_hook)
                )
            elif module == "mlp":
                hooks.append(
                    model.model.layers[i].mlp.register_forward_pre_hook(patch_hook)
                )
            elif module == "attn":
                hooks.append(
                    model.model.layers[i].self_attn.register_forward_pre_hook(
                        patch_hook
                    )
                )
            else:
                raise ValueError("Module %s not supported", module)
        else:
            # when patching a last-layer representation to the last layer of the same
            # model, the final layer norm is not needed because it was already applied
            # (assuming that the representation for patching was obtained by
            # setting output_hidden_representations to True).
            if skip_final_ln and i == len(model.model.layers) - 1 and module == "hs":
                hooks.append(
                    model.model.norm.register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}_skip_ln",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )
            else:
                if module == "hs":
                    hooks.append(model.model.layers[i].register_forward_hook(patch_hook))
                elif module == "mlp":
                    hooks.append(
                        model.model.layers[i].mlp.register_forward_hook(patch_hook)
                    )
                elif module == "attn":
                    hooks.append(
                        model.model.layers[i].self_attn.register_forward_hook(patch_hook)
                    )
                else:
                    raise ValueError("Module %s not supported", module)

    return hooks


def set_hs_patch_hooks_gptj(
        model,
        hs_patch_config,
        module="hs",  # mlp, attn
        patch_input=False,
        skip_final_ln=False,
        generation_mode=False,
):
    """GPTJ patch hooks."""
    # when using mode.generate() the hidden states in the input are cached after
    # the first inference pass, and in the next steps the input/output are of
    # size 1. In these cases we don't need to patch anymore the previous hidden
    # states from the initial input, because they are cached, but we do need
    # to handle these cases in this call because this hook wraps the generation
    # call.
    #
    # NOTE: To use generation mode, we must patch a position that is not the
    # first one. This is because in this case we don't know during generation
    # if we are handling the initial input or a future step and thus don't know
    # if a patching is needed or not.

    # if generation_mode:
    #     for i in hs_patch_config:
    #         for position_, _ in hs_patch_config[i]:
    #             assert position_ > 0

    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, input):
            # (batch, sequence, hidden_state)
            input_len = len(input[0][0])
            if generation_mode and input_len == 1:
                return
            for position_, hs_ in position_hs:
                input[0][0, position_] = hs_

        def post_hook(module, input, output):
            if "skip_ln" in name:
                # output: (batch, sequence, hidden_state)
                output_len = len(output[0])
            else:
                # output[0]: (batch, sequence, hidden_state)
                output_len = len(output[0][0])

            if generation_mode and output_len == 1:
                return
            for position_, hs_ in position_hs:
                if "skip_ln" in name:
                    output[0][position_] = hs_
                else:
                    output[0][0, position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for i in hs_patch_config:
        if patch_input:
            hooks.append(
                model.transformer.h[i].register_forward_pre_hook(
                    patch_hs(
                        f"patch_hs_{i}",
                        hs_patch_config[i],
                        patch_input,
                        generation_mode,
                    )
                )
            )
        else:
            # when patching a last-layer representation to the last layer of the same
            # model, the final layer norm is not needed because it was already applied
            # (assuming that the representation for patching was obtained by
            # setting output_hidden_representations to True).
            if skip_final_ln and i == len(model.transformer.h) - 1:
                hooks.append(
                    model.transformer.ln_f.register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}_skip_ln",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )
            else:
                hooks.append(
                    model.transformer.h[i].register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}",
                            hs_patch_config[i],
                            patch_input,
                            generation_mode,
                        )
                    )
                )

    return hooks


def remove_hooks(hooks):
    if isinstance(hooks, list):
        for hook in hooks:
            hook.remove()
    else:
        hooks.remove()


def unwrap_attn_modules(mt):
    for layer in mt.model.model.layers:
        attn_module = layer.self_attn
        if getattr(attn_module, "_is_wrapped", False):
            attn_module.forward = attn_module._original_forward

            for attr in ["_is_wrapped", "_original_forward", "_cached_hidden_states"]:
                if hasattr(attn_module, attr):
                    delattr(attn_module, attr)

# ##############
#
# Inspection
#
# ##############


def inspect(
        mt,
        prompt_source,
        prompt_target,
        layer_source,
        layer_target,
        position_source,
        position_target,
        module="hs",
        generation_mode=False,
        verbose=False,
        generation_config=None
):
    """Inspection via patching."""
    # adjust position_target to be absolute rather than relative
    inp_target = mt.tokenizer([prompt_target], return_tensors="pt", add_special_tokens=False).to(mt.device)
    if position_target < 0:
        position_target = len(inp_target["input_ids"][0]) + position_target

    # first run the the model on prompt_patch and get all hidden states.
    inp_source = mt.tokenizer([prompt_source], return_tensors="pt", add_special_tokens=False).to(mt.device)
    if verbose:
        print(
            "prompt_patch:",
            [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]],
        )

    hs_cache_ = []
    # We manually store intermediate states that the model API does not expose
    store_hooks = []
    if module == "mlp":

        def store_mlp_hook(module, input, output):
            hs_cache_.append(output[0])

        for layer in mt.model.model.layers:
            store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
    elif module == "attn":

        def store_attn_hook(module, input, output):
            hs_cache_.append(output[0].squeeze())

        for layer in mt.model.model.layers:
            store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

    output = mt.model(**inp_source, output_hidden_states=True)
    if module == "hs":
        hs_cache_ = [
            output["hidden_states"][layer + 1][0] for layer in range(mt.num_layers)
        ]

    remove_hooks(store_hooks)
    # now do a second run on prompt, while patching
    # a specific hidden state from the first run.
    hs_patch_config = {
        layer_target: [(
            position_target,
            hs_cache_[layer_source][position_source],
        )]
    }

    if layer_source == layer_target == mt.num_layers - 1:
        skip_final_ln = True
    else:
        skip_final_ln = False
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        skip_final_ln=skip_final_ln,
        generation_mode=True,
    )

    # Single prediction / generation
    if verbose:
        print(
            "prompt:", [mt.tokenizer.decode(x) for x in inp_source["input_ids"][0]]
        )
        print(
            f"patching position {position_target} with the hidden state from layer"
            f" {layer_source} at position {position_source}."
        )
    if generation_mode:
        # Checking if should perform temperature sampling, to allow smoother
        # non-repeating long outputs.
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            tokenizer=mt.tokenizer,
            generation_config=generation_config
        )[0][len(inp_target["input_ids"][0]):]

        output = mt.tokenizer.decode(output_toks)
        if verbose:
            print(
                "generation with patching: ",
                [mt.tokenizer.decode(x) for x in output_toks],
            )
    else:
        output = mt.model(**inp_target)
        answer_prob, answer_t = torch.max(
            torch.softmax(output.logits[0, -1, :], dim=0), dim=0
        )
        output = decode_tokens(mt.tokenizer, [answer_t])[0], round(
            answer_prob.cpu().item(), 4
        )
        if verbose:
            print("prediction with patching: ", output)

    # remove patching hooks
    remove_hooks(patch_hooks)

    return output


def get_activations(
        mt,
        inputs,
        position,
        module="hs",
):
    """Get activations."""
    hs_cache_ = []
    # We manually store intermediate states that the model API does not expose
    store_hooks = []
    if module == "mlp":
        raise NotImplementedError
        def store_mlp_hook(module, input, output):
            hs_cache_.append(output[0, position])

        for layer in mt.model.model.layers:
            store_hooks.append(layer.mlp.register_forward_hook(store_mlp_hook))
    elif module == "attn":
        raise NotImplementedError
        def store_attn_hook(module, input, output):
            hs_cache_.append(output[0, position].squeeze())

        for layer in mt.model.model.layers:
            store_hooks.append(layer.self_attn.register_forward_hook(store_attn_hook))

    if module == "hs":
        output = mt.model(**inputs, output_hidden_states=True)
        hs_cache_ = [
            output["hidden_states"][layer + 1][0, position] for layer in range(mt.num_layers)
        ]
    elif module == 'attn_weights':
        output = mt.model(**inputs, output_attentions=True)
        hs_cache_ = [
            output["attentions"][layer][0, :, position] for layer in range(mt.num_layers)  # [bsz, num_heads, query, key]
        ]
    else:
        output = mt.model(**inputs)

    # save activations
    hs_cache_ = torch.stack(hs_cache_, dim=0)  # hs: [num_layers, hidden_dim] / attn_weights: [num_layers, num_heads, seq_len]

    remove_hooks(store_hooks)

    return hs_cache_


from transformers.models.llama.modeling_llama import repeat_kv
def get_attentions(
        mt,
        inputs,
        position,
        module="attn",
):
    """Get activations."""
    assert module == 'attn' or module == 'attn_weights'
    attentions = []
    # We manually store intermediate states that the model API does not expose
    store_hooks = []

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

    def store_attn(start_position):
        def store_attn_hook(module, input, output):
            # query token i (position), key/value token j
            hidden_states = module._cached_hidden_states
            bsz, q_len, hidden_dim = hidden_states.size()
            assert bsz == 1, "This hook only supports batch size = 1"
            num_tokens = q_len - start_position

            num_key_value_heads = module.num_key_value_heads
            head_dim = module.head_dim

            value_states = module.v_proj(hidden_states)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1,
                                                                                                  2)  # [bsz, num_heads, seq_len, head_dim]
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
            attentions.append(a_i_norm.clone().detach())

        return store_attn_hook

    if module == 'attn':
        for layer in mt.model.model.layers:
            wrap_forward_with_cache_input(layer.self_attn)
            store_hooks.append(layer.self_attn.register_forward_hook(store_attn(position)))

    output = mt.model(**inputs, output_attentions=True)
    attention_weights = [
        output["attentions"][layer][0, :, position:] for layer in range(mt.num_layers)  # [bsz, num_heads, query, key]
    ]

    # save activations
    if module == 'attn':
        attentions = torch.stack(attentions, dim=0)  # [num_layers, num_tokens, seq_len]
    else:
        attentions = None
    attention_weights = torch.stack(attention_weights, dim=0)  # [num_layers, num_heads, num_tokens, seq_len]

    remove_hooks(store_hooks)
    unwrap_attn_modules(mt)

    return {'attentions': attentions, 'attention_weights': attention_weights}


def get_activations_while_generate(
        mt,
        inputs,
        max_new_tokens,
        module="hs",
):
    """Get activations."""
    hs_cache_ = []
    # We manually store intermediate states that the model API does not expose
    store_hooks = []
    assert module == 'hs'

    generated_ids = inputs['input_ids']
    past_key_values = None

    for _ in range(max_new_tokens):
        hs_cache_.append([])
        output = mt.model(input_ids=generated_ids,
                          past_key_values=past_key_values,
                          output_hidden_states=True,
                          use_cache=True)
        if module == "hs":
            hs_cache_[-1] = torch.stack([
                output["hidden_states"][layer + 1][0, -1] for layer in range(mt.num_layers)
            ])

        past_key_values = output.past_key_values
        next_token_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == mt.tokenizer.eos_token_id:
            break

    # save activations
    hs_cache_ = torch.stack(hs_cache_, dim=0)  # [num_layers, hidden_dim]

    remove_hooks(store_hooks)

    return hs_cache_, generated_ids  # [num_generated_tokens, num_layers, hidden_dim]


def patch(
        mt,
        prompt_source,
        prompt_target,
        patch_layers,
        position_source,
        position_target,
        module="hs",
        generation_mode=False,
        verbose=False,
        generation_config=None
):
    """Inspection via patching."""
    # adjust position_target to be absolute rather than relative
    inp_target = mt.tokenizer([prompt_target], return_tensors="pt", add_special_tokens=False).to(mt.device)
    if position_target < 0:
        position_target = len(inp_target["input_ids"][0]) + position_target

    # first run the the model on prompt_patch and get all hidden states.
    inp_source = mt.tokenizer([prompt_source], return_tensors="pt", add_special_tokens=False).to(mt.device)

    hs_cache_ = []
    # We manually store intermediate states that the model API does not expose
    store_hooks = []
    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    output = mt.model(**inp_source, output_hidden_states=True)
    if module == "hs":
        hs_cache_ = [
            output["hidden_states"][layer + 1][0] for layer in range(mt.num_layers)
        ]

    remove_hooks(store_hooks)
    # now do a second run on prompt, while patching
    # a specific hidden state from the first run.
    hs_patch_config = {}
    for layer in patch_layers:
        hs_patch_config[layer] = [(position_target, hs_cache_[layer][position_source])]

    assert mt.num_layers - 1 not in patch_layers
    skip_final_ln = False
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        skip_final_ln=skip_final_ln,
        generation_mode=True,
    )

    if generation_mode:
        # Checking if should perform temperature sampling, to allow smoother
        # non-repeating long outputs.
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            tokenizer=mt.tokenizer,
            generation_config=generation_config
        )[0][len(inp_target["input_ids"][0]):]

        output = mt.tokenizer.decode(output_toks)
        if verbose:
            print(
                "generation with patching: ",
                [mt.tokenizer.decode(x) for x in output_toks],
            )
    else:
        output = mt.model(**inp_target)
        answer_prob, answer_t = torch.max(
            torch.softmax(output.logits[0, -1, :], dim=0), dim=0
        )
        output = decode_tokens(mt.tokenizer, [answer_t])[0], round(
            answer_prob.cpu().item(), 4
        )
        if verbose:
            print("prediction with patching: ", output)

    # remove patching hooks
    remove_hooks(patch_hooks)

    return output


def evaluate_patch_next_token_prediction(
        mt,
        prompt_source,
        prompt_target,
        layer_source,
        layer_target,
        position_source,
        position_target,
        module="hs",
        position_prediction=-1,
        transform=None,
):
    """Evaluate next token prediction."""
    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    # adjust position_target to be absolute rather than relative
    inp_target = mt.tokenizer([prompt_target], return_tensors="pt", add_special_tokens=False).to(mt.device)
    if position_target < 0:
        position_target = len(inp_target["input_ids"][0]) + position_target

    # first run the the model on without patching and get the results.
    inp_source = mt.tokenizer([prompt_source], return_tensors="pt", add_special_tokens=False).to(mt.device)
    output_orig = mt.model(**inp_source, output_hidden_states=True)
    dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
    _, answer_t_orig = torch.max(dist_orig, dim=0)
    hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
        position_source
    ]
    if transform is not None:
        hidden_rep = transform(hidden_rep)

    # now do a second run on prompt, while patching the input hidden state.
    hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
    if layer_source == layer_target == mt.num_layers - 1:
        skip_final_ln = True
    else:
        skip_final_ln = False
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        skip_final_ln=skip_final_ln,
        generation_mode=True,
    )
    output = mt.model(**inp_target)
    dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
    _, answer_t = torch.max(dist, dim=0)

    # remove patching hooks
    remove_hooks(patch_hooks)

    prec_1 = (answer_t == answer_t_orig).detach().cpu().item()
    surprisal = -torch.log(dist_orig[answer_t]).detach().cpu().numpy()

    return prec_1, surprisal


def evaluate_patch_next_token_prediction_x_model(
        mt_1,
        mt_2,
        prompt_source,
        prompt_target,
        layer_source,
        layer_target,
        position_source,
        position_target,
        module="hs",
        position_prediction=-1,
        transform=None,
):
    """evaluate next token prediction across models."""
    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    # adjust position_target to be absolute rather than relative
    inp_target = make_inputs(mt_2.tokenizer, [prompt_target], device=mt_2.device)
    if position_target < 0:
        position_target = len(inp_target["input_ids"][0]) + position_target

    # first run the the model on without patching and get the results.
    inp_source = make_inputs(mt_1.tokenizer, [prompt_source], device=mt_1.device)
    output_orig = mt_1.model(**inp_source, output_hidden_states=True)
    dist_orig = torch.softmax(output_orig.logits[0, position_source, :], dim=0)
    _, answer_t_orig = torch.max(dist_orig, dim=0)
    hidden_rep = output_orig["hidden_states"][layer_source + 1][0][
        position_source
    ]
    if transform is not None:
        hidden_rep = transform(hidden_rep)

    # now do a second run on prompt, while patching the input hidden state.
    hs_patch_config = {layer_target: [(position_target, hidden_rep)]}
    skip_final_ln = False
    patch_hooks = mt_2.set_hs_patch_hooks(
        mt_2.model,
        hs_patch_config,
        module=module,
        patch_input=False,
        skip_final_ln=skip_final_ln,
        generation_mode=True,
    )
    output = mt_2.model(**inp_target)
    dist = torch.softmax(output.logits[0, position_prediction, :], dim=0)
    _, answer_t = torch.max(dist, dim=0)

    # remove patching hooks
    remove_hooks(patch_hooks)

    prec_1 = answer_t.detach().cpu().item() == answer_t_orig.detach().cpu().item()
    surprisal = -torch.log(dist_orig[answer_t]).detach().cpu().numpy()

    return prec_1, surprisal


# Adding support for batched patching. More than 10x speedup
# Currently only supporting GPT-J
def set_hs_patch_hooks_gptj_batch(
        model,
        hs_patch_config,
        module="hs",
        patch_input=False,
        generation_mode=False,
):
    """GPTJ patch hooks - supporting batch."""
    # when using mode.generate() the hidden states in the input are cached after
    # the first inference pass, and in the next steps the input/output are of
    # size 1. In these cases we don't need to patch anymore the previous hidden
    # states from the initial input, because they are cached, but we do need to
    # handle these cases in this call because this hook wraps the generation call.
    #
    # NOTE: To use generation mode, we must patch a position that is not the
    # first one. This is because in this case we don't know during generation if
    # we are handling the initial input or a future step and thus don't know if
    # a patching is needed or not.

    # if generation_mode:
    #     for i in hs_patch_config:
    #         for position_, _ in hs_patch_config[i]:
    #             assert position_ > 0

    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, inp):
            # (batch, sequence, hidden_state)
            idx_, position_, hs_ = (
                position_hs["batch_idx"],
                position_hs["position_target"],
                position_hs["hidden_rep"],
            )
            input_len = len(inp[0][idx_])
            if generation_mode and input_len == 1:
                return
            inp[0][idx_][position_] = hs_

        def post_hook(module, inp, output):
            idx_, position_, hs_ = (
                position_hs["batch_idx"],
                position_hs["position_target"],
                position_hs["hidden_rep"],
            )
            if "skip_ln" in name:
                # output: (batch, sequence, hidden_state)
                output_len = len(output[idx_])
                if generation_mode and output_len == 1:
                    return
                output[idx_][position_] = hs_
            else:
                # output[0]: (batch, sequence, hidden_state)
                output_len = len(output[0][idx_])
                if generation_mode and output_len == 1:
                    return
                output[0][idx_][position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []
    for item in hs_patch_config:
        i = item["layer_target"]
        skip_final_ln = item["skip_final_ln"]
        if patch_input:
            hooks.append(
                model.transformer.h[i].register_forward_pre_hook(
                    patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
                )
            )
        else:
            # when patching a last-layer representation to the last layer of the same
            # model, the final layer norm is not needed because it was already
            # applied (assuming that the representation for patching was obtained by
            # setting output_hidden_representations to True).
            if skip_final_ln and i == len(model.transformer.h) - 1:
                hooks.append(
                    model.transformer.ln_f.register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}_skip_ln",
                            item,
                            patch_input,
                            generation_mode,
                        )
                    )
                )
            else:
                hooks.append(
                    model.transformer.h[i].register_forward_hook(
                        patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
                    )
                )

    return hooks


def set_hs_patch_hooks_llama_batch(
        model,
        hs_patch_config,
        module="hs",
        patch_input=False,
        generation_mode=False,
):
    """LLAMA patch hooks - supporting batch."""
    # when using mode.generate() the hidden states in the input are cached after
    # the first inference pass, and in the next steps the input/output are of
    # size 1. In these cases we don't need to patch anymore the previous hidden
    # states from the initial input, because they are cached, but we do need to
    # handle these cases in this call because this hook wraps the generation call.
    #
    # NOTE: To use generation mode, we must patch a position that is not the
    # first one. This is because in this case we don't know during generation if
    # we are handling the initial input or a future step and thus don't know if
    # a patching is needed or not.

    # if generation_mode:
    #     for i in hs_patch_config:
    #         for position_, _ in hs_patch_config[i]:
    #             assert position_ > 0

    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    def patch_hs(name, position_hs, patch_input, generation_mode):
        def pre_hook(module, inp):
            # inp[0]: (batch, sequence, hidden_state)
            idx_, position_, hs_ = (
                position_hs["batch_idx"],
                position_hs["position_target"],
                position_hs["hidden_rep"],
            )
            input_len = len(inp[0][idx_])
            if generation_mode and input_len == 1:
                return
            inp[0][idx_][position_] = hs_

        def post_hook(module, inp, output):
            idx_, position_, hs_ = (
                position_hs["batch_idx"],
                position_hs["position_target"],
                position_hs["hidden_rep"],
            )
            if "skip_ln" in name:
                # output: (batch, sequence, hidden_state)
                output_len = len(output[idx_])
                if generation_mode and output_len == 1:
                    return
                output[idx_][position_] = hs_
            else:
                # output[0]: (batch, sequence, hidden_state)
                output_len = len(output[0][idx_])
                if generation_mode and output_len == 1:
                    return
                output[0][idx_][position_] = hs_

        if patch_input:
            return pre_hook
        else:
            return post_hook

    hooks = []

    for item in hs_patch_config:
        i = item["layer_target"]
        skip_final_ln = item["skip_final_ln"]
        if patch_input:
            hooks.append(
                model.model.layers[i].register_forward_pre_hook(
                    patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
                )
            )
        else:
            # when patching a last-layer representation to the last layer of the same
            # model, the final layer norm is not needed because it was already applied
            # (assuming that the representation for patching was obtained by setting
            # output_hidden_representations to True).
            if skip_final_ln and i == len(model.model.layers) - 1:
                hooks.append(
                    model.model.norm.register_forward_hook(
                        patch_hs(
                            f"patch_hs_{i}_skip_ln", item, patch_input, generation_mode
                        )
                    )
                )
            else:
                hooks.append(
                    model.model.layers[i].register_forward_hook(
                        patch_hs(f"patch_hs_{i}", item, patch_input, generation_mode)
                    )
                )

    return hooks


def evaluate_patch_next_token_prediction_batch(
        mt, df, batch_size=256, transform=None, module="hs"
):
    """Evaluate next token prediction with batch support."""
    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    prec_1 = np.zeros(0)
    surprisal = np.zeros(0)
    next_token = np.zeros(0)

    #     generations = []

    def _evaluat_single_batch(batch_df):
        batch_size = len(batch_df)
        prompt_source_batch = np.array(batch_df["prompt_source"])
        prompt_target_batch = np.array(batch_df["prompt_target"])
        layer_source_batch = np.array(batch_df["layer_source"])
        layer_target_batch = np.array(batch_df["layer_target"])
        position_source_batch = np.array(batch_df["position_source"])
        position_target_batch = np.array(batch_df["position_target"])
        position_prediction_batch = np.ones_like(position_target_batch) * -1
        #         max_gen_len = np.array(batch_df["max_gen_len"])

        # adjust position_target to be absolute rather than relative
        inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
        for i in range(batch_size):
            if position_target_batch[i] < 0:
                position_target_batch[i] += len(inp_target["input_ids"][i])

        # first run the the model on without patching and get the results.
        inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
        output_orig = mt.model(**inp_source, output_hidden_states=True)
        dist_orig = torch.softmax(
            output_orig.logits[
            np.array(range(batch_size)), position_source_batch, :
            ],
            dim=-1,
        )
        _, answer_t_orig = torch.max(dist_orig, dim=-1)
        # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
        hidden_rep = [
            output_orig.hidden_states[layer_source_batch[i] + 1][i][
                position_source_batch[i]
            ]
            for i in range(batch_size)
        ]
        if transform is not None:
            for i in range(batch_size):
                hidden_rep[i] = transform(hidden_rep[i])

        # now do a second run on prompt, while patching the input hidden state.
        hs_patch_config = [
            {
                "batch_idx": i,
                "layer_target": layer_target_batch[i],
                "position_target": position_target_batch[i],
                "hidden_rep": hidden_rep[i],
                "skip_final_ln": (
                        layer_source_batch[i]
                        == layer_target_batch[i]
                        == mt.num_layers - 1
                ),
            }
            for i in range(batch_size)
        ]
        patch_hooks = mt.set_hs_patch_hooks(
            mt.model,
            hs_patch_config,
            module=module,
            patch_input=False,
            generation_mode=False,
        )

        output = mt.model(**inp_target)

        # # NOTE: inputs are left padded,
        # # and sequence length is the same across batch
        # # to support generations of variable lengths,
        # # first generate with maximum number of tokens needed in the batch
        # seq_len = len(inp_target["input_ids"][0])
        # output_toks = mt.model.generate(
        #     inp_target["input_ids"],
        #     max_length=seq_len + max(max_gen_len),
        #     pad_token_id=mt.model.generation_config.eos_token_id,
        # )[:, seq_len:]

        # # then, we select only the subset of tokens that we need
        # generations = [
        #     mt.tokenizer.decode(output_toks[i][: max_gen_len[i]])
        #     for i in range(batch_size)
        # ]

        dist = torch.softmax(
            output.logits[
            np.array(range(batch_size)), position_prediction_batch, :
            ],
            dim=-1,
        )
        _, answer_t = torch.max(dist, dim=-1)
        next_token = [mt.tokenizer.decode(tok) for tok in answer_t]

        # remove patching hooks
        remove_hooks(patch_hooks)

        prec_1 = (answer_t == answer_t_orig).detach().cpu().numpy()
        surprisal = (
            -torch.log(dist_orig[np.array(range(batch_size)), answer_t])
            .detach()
            .cpu()
            .numpy()
        )

        return prec_1, surprisal, next_token

    for i in tqdm.tqdm(range(len(df) // batch_size)):
        cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
        batch_prec_1, batch_surprisal, batch_next_token = _evaluat_single_batch(
            cur_df
        )
        prec_1 = np.concatenate((prec_1, batch_prec_1))
        surprisal = np.concatenate((surprisal, batch_surprisal))
        next_token = np.concatenate((next_token, batch_next_token))

    return prec_1, surprisal, next_token


def inspect_batch(mt, df, batch_size=256, transform=None, module="hs"):
    """Inspects batch: source/target layer/position could differ within batch."""
    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    generations = []

    def _inspect_single_batch(batch_df):
        batch_size = len(batch_df)
        prompt_source_batch = np.array(batch_df["prompt_source"])
        prompt_target_batch = np.array(batch_df["prompt_target"])
        layer_source_batch = np.array(batch_df["layer_source"])
        layer_target_batch = np.array(batch_df["layer_target"])
        position_source_batch = np.array(batch_df["position_source"])
        position_target_batch = np.array(batch_df["position_target"])
        max_gen_len = np.array(batch_df["max_gen_len"])

        # adjust position_target to be absolute rather than relative
        inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
        for i in range(batch_size):
            if position_target_batch[i] < 0:
                position_target_batch[i] += len(inp_target["input_ids"][i])

        # first run the the model on without patching and get the results.
        inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
        output_orig = mt.model(**inp_source, output_hidden_states=True)

        # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
        hidden_rep = [
            output_orig.hidden_states[layer_source_batch[i] + 1][i][
                position_source_batch[i]
            ]
            for i in range(batch_size)
        ]
        if transform is not None:
            for i in range(batch_size):
                hidden_rep[i] = transform(hidden_rep[i])

        # now do a second run on prompt, while patching the input hidden state.
        hs_patch_config = [
            {
                "batch_idx": i,
                "layer_target": layer_target_batch[i],
                "position_target": position_target_batch[i],
                "hidden_rep": hidden_rep[i],
                "skip_final_ln": (
                        layer_source_batch[i]
                        == layer_target_batch[i]
                        == mt.num_layers - 1
                ),
            }
            for i in range(batch_size)
        ]
        patch_hooks = mt.set_hs_patch_hooks(
            mt.model,
            hs_patch_config,
            module=module,
            patch_input=False,
            generation_mode=True,
        )

        # NOTE: inputs are left padded,
        # and sequence length is the same across batch
        # to support generations of variable lengths,
        # first generate with maximum number of tokens needed in the batch
        seq_len = len(inp_target["input_ids"][0])
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=seq_len + max(max_gen_len),
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[:, seq_len:]

        # then, we select only the subset of tokens that we need
        generations = [
            mt.tokenizer.decode(output_toks[i][: max_gen_len[i]])
            for i in range(batch_size)
        ]

        # remove patching hooks
        remove_hooks(patch_hooks)

        return generations

    for i in tqdm.tqdm(range(1 + len(df) // batch_size)):
        cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
        batch_generations = _inspect_single_batch(cur_df)
        generations.extend(batch_generations)

    return generations


def evaluate_attriburte_exraction_batch(
        mt,
        df,
        batch_size=256,
        max_gen_len=10,
        transform=None,
        is_icl=True,
        module="hs",
):
    """Evaluates attribute extraction with batch support."""
    # We don't know the exact token position of the
    # attribute, as it is not necessarily the next token. So, precision and
    # surprisal may not apply directly.

    if module != "hs":
        raise ValueError("Module %s not yet supported", module)

    def _evaluate_attriburte_exraction_single_batch(batch_df):
        batch_size = len(batch_df)
        prompt_source_batch = np.array(batch_df["prompt_source"])
        prompt_target_batch = np.array(batch_df["prompt_target"])
        layer_source_batch = np.array(batch_df["layer_source"])
        layer_target_batch = np.array(batch_df["layer_target"])
        position_source_batch = np.array(batch_df["position_source"])
        position_target_batch = np.array(batch_df["position_target"])

        object_batch = np.array(batch_df["object"])

        # Adjust position_target to be absolute rather than relative
        inp_target = make_inputs(mt.tokenizer, prompt_target_batch, mt.device)
        for i in range(batch_size):
            if position_target_batch[i] < 0:
                position_target_batch[i] += len(inp_target["input_ids"][i])

        # Step 1: run model on source prompt without patching and get the hidden
        # representations.
        inp_source = make_inputs(mt.tokenizer, prompt_source_batch, mt.device)
        output_orig = mt.model(**inp_source, output_hidden_states=True)

        # hidden_states size (n_layers, n_sample, seq_len, hidden_dim)
        #         hidden_rep = []
        #         for i in range(batch_size):
        #             hidden_rep.append(output_orig.hidden_states[layer_source_batch[i] + 1][i][position_source_batch[i]])
        hidden_rep = [
            output_orig.hidden_states[layer_source_batch[i] + 1][i][
                position_source_batch[i]
            ]
            for i in range(batch_size)
        ]
        if transform is not None:
            for i in range(batch_size):
                hidden_rep[i] = transform(hidden_rep[i])

        # Step 2: Do second run on target prompt, while patching the input
        # hidden state.
        hs_patch_config = [
            {
                "batch_idx": i,
                "layer_target": layer_target_batch[i],
                "position_target": position_target_batch[i],
                "hidden_rep": hidden_rep[i],
                "skip_final_ln": (
                        layer_source_batch[i]
                        == layer_target_batch[i]
                        == mt.num_layers - 1
                ),
            }
            for i in range(batch_size)
        ]
        patch_hooks = mt.set_hs_patch_hooks(
            mt.model,
            hs_patch_config,
            module=module,
            patch_input=False,
            generation_mode=True,
        )

        # Note that inputs are left padded,
        # and sequence length is the same across batch
        seq_len = len(inp_target["input_ids"][0])
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=seq_len + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[:, seq_len:]
        generations_patched = decode_tokens(mt.tokenizer, output_toks)
        if is_icl:
            prefix = batch_df["prefix"].iloc[0]

            def _crop_by_prefix(generations, prefix):
                concatenated_str = " ".join(generations)
                _pos = concatenated_str.find(prefix)
                return concatenated_str[:_pos]

            generations_patched_postprocessed = np.array([
                _crop_by_prefix(generations_patched[i], prefix)
                for i in range(batch_size)
            ])
        else:
            generations_patched_postprocessed = np.array(
                [" ".join(generations_patched[i]) for i in range(batch_size)]
            )

        is_correct_patched = np.array([
            object_batch[i].replace(" ", "")
            in generations_patched_postprocessed[i].replace(" ", "")
            for i in range(batch_size)
        ])

        # remove patching hooks
        remove_hooks(patch_hooks)

        cpu_hidden_rep = np.array(
            [hidden_rep[i].detach().cpu().numpy() for i in range(batch_size)]
        )

        results = {
            "generations_patched": generations_patched,
            "generations_patched_postprocessed": generations_patched_postprocessed,
            "is_correct_patched": is_correct_patched,
            "hidden_rep": cpu_hidden_rep,
        }

        return results

    results = {}
    n_batches = len(df) // batch_size
    if len(df) % batch_size != 0:
        n_batches += 1
    for i in tqdm(range(len(df) // batch_size)):
        cur_df = df.iloc[batch_size * i: batch_size * (i + 1)]
        batch_results = _evaluate_attriburte_exraction_single_batch(cur_df)
        for key, value in batch_results.items():
            if key in results:
                results[key] = np.concatenate((results[key], value))
            else:
                results[key] = value

    return results
