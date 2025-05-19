import os
import json
import h5py
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt


def find_offsets(s, t, offsets):
    assert t in s
    start_char = s.rfind(t)
    end_char = start_char + len(t)

    token_indices = [
        i for i, (start, end) in enumerate(offsets)
        if start < end_char and end > start_char
    ]
    return token_indices[0], token_indices[-1]


def intervention_impact_attn_weights_wikidata(input_filename, intervention_filename, tokenizer_path, token_position=-1, agg='sum', save_heads_path=None, save_heads=False):
    data = [json.loads(line) for line in open(input_filename)]
    interv_data = [json.loads(line) for line in open(intervention_filename)]
    assert len(data) == len(interv_data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
        num_layers, num_heads, num_tokens, max_seq_len = f['attn_weights'][0].shape  # [num_layers, num_heads, num_tokens, seq_len]

    topk_heads = None
    if not save_heads and save_heads_path is not None:
        topk_heads = [json.loads(line) for line in open(save_heads_path)][:48]

    profession_decrease_sum = np.zeros((num_layers, num_heads))
    city_decrease_sum = np.zeros((num_layers, num_heads))
    name_increase_sum = np.zeros((num_layers, num_heads))

    total = 0
    for index, (instance, interv_instance) in enumerate(zip(data, interv_data)):
        # if not (instance['but_tag'] and not interv_instance['but_tag']):
        #     continue
        assert instance['prompt'] == interv_instance['prompt']
        total += 1

        encoding = tokenizer(instance['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        q_len = len(tokens)

        offsets = encoding['offset_mapping']
        profession_offsets = find_offsets(instance['prompt'], instance['profession'], offsets)
        city_offsets = find_offsets(instance['prompt'], instance['city'], offsets)
        name_offsets = find_offsets(instance['prompt'], instance['name'], offsets)

        with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
            attn_weights = f['attn_weights'][index][:, :, token_position]  # [num_layers, num_heads, seq_len]

        with h5py.File(os.path.join(os.path.dirname(intervention_filename), 'attn_weights.h5'), 'r') as f:
            interv_attn_weights = f['attn_weights'][index][:, :, token_position]

        profession_attn_weights = attn_weights[:, :, profession_offsets[0]:profession_offsets[1] + 1]
        city_attn_weights = attn_weights[:, :, city_offsets[0]:city_offsets[1] + 1]
        name_attn_weights = attn_weights[:, :, name_offsets[0]:name_offsets[1] + 1]

        profession_interv_attn_weights = interv_attn_weights[:, :, profession_offsets[0]:profession_offsets[1] + 1]
        city_interv_attn_weights = interv_attn_weights[:, :, city_offsets[0]:city_offsets[1] + 1]
        name_interv_attn_weights = interv_attn_weights[:, :, name_offsets[0]:name_offsets[1] + 1]

        if agg == 'max':
            profession_attn = np.max(profession_attn_weights, axis=2)
            city_attn = np.max(city_attn_weights, axis=2)
            name_attn = np.max(name_attn_weights, axis=2)
            profession_interv_attn = np.max(profession_interv_attn_weights, axis=2)
            city_interv_attn = np.max(city_interv_attn_weights, axis=2)
            name_interv_attn = np.max(name_interv_attn_weights, axis=2)
        elif agg == 'mean':
            profession_attn = np.mean(profession_attn_weights, axis=2)
            city_attn = np.mean(city_attn_weights, axis=2)
            name_attn = np.mean(name_attn_weights, axis=2)
            profession_interv_attn = np.mean(profession_interv_attn_weights, axis=2)
            city_interv_attn = np.mean(city_interv_attn_weights, axis=2)
            name_interv_attn = np.mean(name_interv_attn_weights, axis=2)
        elif agg == 'sum':
            profession_attn = np.sum(profession_attn_weights, axis=2)
            city_attn = np.sum(city_attn_weights, axis=2)
            name_attn = np.sum(name_attn_weights, axis=2)
            profession_interv_attn = np.sum(profession_interv_attn_weights, axis=2)
            city_interv_attn = np.sum(city_interv_attn_weights, axis=2)
            name_interv_attn = np.sum(name_interv_attn_weights, axis=2)
        else:
            raise NotImplementedError

        profession_decrease_sum += (profession_attn - profession_interv_attn)
        city_decrease_sum += (city_attn - city_interv_attn)
        name_increase_sum += (name_interv_attn - name_attn)

    profession_decrease_avg = profession_decrease_sum / total
    city_decrease_avg = city_decrease_sum / total
    name_increase_avg = name_increase_sum / total

    print(f'\nMean Δ attention (profession decrease): {np.mean(profession_decrease_avg):.4f}')
    print(f'Mean Δ attention (city decrease): {np.mean(city_decrease_avg):.4f}')
    print(f'Mean Δ attention (name increase): {np.mean(name_increase_avg):.4f}')

    if topk_heads is not None:
        topk_profession_decrease_avg = [profession_decrease_avg[h['layer'], h['head']] for h in topk_heads]
        topk_city_decrease_avg = [city_decrease_avg[h['layer'], h['head']] for h in topk_heads]
        topk_name_increase_avg = [name_increase_avg[h['layer'], h['head']] for h in topk_heads]
        print(f'Mean Δ attention (profession decrease) for top heads: {np.mean(topk_profession_decrease_avg):.4f}')
        print(f'Mean Δ attention (city decrease) for top heads: {np.mean(topk_city_decrease_avg):.4f}')
        print(f'Mean Δ attention (name increase) for top heads: {np.mean(topk_name_increase_avg):.4f}')

    combined_score = name_increase_avg  # + city_decrease_avg
    flat_scores = combined_score.flatten()
    num_layers, num_heads = combined_score.shape

    top_n = 100
    top_indices = np.argsort(flat_scores)[-top_n:][::-1]

    if save_heads:
        assert save_heads_path is not None
        if not os.path.exists(os.path.dirname(save_heads_path)):
            os.makedirs(os.path.dirname(save_heads_path), exist_ok=True)
        with open(save_heads_path, 'w') as f_out:
            for idx in top_indices:
                layer = int(idx // num_heads)
                head = int(idx % num_heads)
                result = {
                    'layer': layer,
                    'head': head
                }
                f_out.write(json.dumps(result) + '\n')


def intervention_impact_attn_weights_celebrity(input_filename, intervention_filename, tokenizer_path, token_position=-1, agg='sum', save_heads_path=None, save_heads=False):
    data = [json.loads(line) for line in open(input_filename)]
    interv_data = [json.loads(line) for line in open(intervention_filename)]
    assert len(data) == len(interv_data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
        num_layers, num_heads, num_tokens, max_seq_len = f['attn_weights'][0].shape  # [num_layers, num_heads, num_tokens, seq_len]

    topk_heads = None
    if not save_heads and save_heads_path is not None:
        topk_heads = [json.loads(line) for line in open(save_heads_path)][:48]

    parent_decrease_sum = np.zeros((num_layers, num_heads))
    name_increase_sum = np.zeros((num_layers, num_heads))

    total = 0
    for index, (instance, interv_instance) in enumerate(zip(data, interv_data)):
        # if not (instance['but_tag'] and not interv_instance['but_tag']):
        #     continue
        assert instance['prompt'] == interv_instance['prompt']
        total += 1

        encoding = tokenizer(instance['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        q_len = len(tokens)

        offsets = encoding['offset_mapping']
        parent_offsets = find_offsets(instance['prompt'], instance['parent_name'], offsets)
        name_offsets = find_offsets(instance['prompt'], instance['name'], offsets)

        with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
            attn_weights = f['attn_weights'][index][:, :, token_position]  # [num_layers, num_heads, seq_len]

        with h5py.File(os.path.join(os.path.dirname(intervention_filename), 'attn_weights.h5'), 'r') as f:
            interv_attn_weights = f['attn_weights'][index][:, :, token_position]

        parent_attn_weights = attn_weights[:, :, parent_offsets[0]:parent_offsets[1] + 1]
        name_attn_weights = attn_weights[:, :, name_offsets[0]:name_offsets[1] + 1]

        parent_interv_attn_weights = interv_attn_weights[:, :, parent_offsets[0]:parent_offsets[1] + 1]
        name_interv_attn_weights = interv_attn_weights[:, :, name_offsets[0]:name_offsets[1] + 1]

        if agg == 'max':
            parent_attn = np.max(parent_attn_weights, axis=2)
            name_attn = np.max(name_attn_weights, axis=2)
            parent_interv_attn = np.max(parent_interv_attn_weights, axis=2)
            name_interv_attn = np.max(name_interv_attn_weights, axis=2)
        elif agg == 'mean':
            parent_attn = np.mean(parent_attn_weights, axis=2)
            name_attn = np.mean(name_attn_weights, axis=2)
            parent_interv_attn = np.mean(parent_interv_attn_weights, axis=2)
            name_interv_attn = np.mean(name_interv_attn_weights, axis=2)
        elif agg == 'sum':
            parent_attn = np.sum(parent_attn_weights, axis=2)
            name_attn = np.sum(name_attn_weights, axis=2)
            parent_interv_attn = np.sum(parent_interv_attn_weights, axis=2)
            name_interv_attn = np.sum(name_interv_attn_weights, axis=2)
        else:
            raise NotImplementedError

        parent_decrease_sum += (parent_attn - parent_interv_attn)
        name_increase_sum += (name_interv_attn - name_attn)

    parent_decrease_avg = parent_decrease_sum / total
    name_increase_avg = name_increase_sum / total

    print(f'\nMean Δ attention (parent decrease): {np.mean(parent_decrease_avg):.4f}')
    print(f'Mean Δ attention (name increase): {np.mean(name_increase_avg):.4f}')

    if topk_heads is not None:
        topk_parent_decrease_avg = [parent_decrease_avg[h['layer'], h['head']] for h in topk_heads]
        topk_name_increase_avg = [name_increase_avg[h['layer'], h['head']] for h in topk_heads]
        print(f'Mean Δ attention (parent decrease) for top heads: {np.mean(topk_parent_decrease_avg):.4f}')
        print(f'Mean Δ attention (name increase) for top heads: {np.mean(topk_name_increase_avg):.4f}')

    combined_score = name_increase_avg  # + city_decrease_avg
    flat_scores = combined_score.flatten()
    num_layers, num_heads = combined_score.shape

    top_n = 100
    top_indices = np.argsort(flat_scores)[-top_n:][::-1]

    if save_heads:
        assert save_heads_path is not None
        if not os.path.exists(os.path.dirname(save_heads_path)):
            os.makedirs(os.path.dirname(save_heads_path), exist_ok=True)
        with open(save_heads_path, 'w') as f_out:
            for idx in top_indices:
                layer = int(idx // num_heads)
                head = int(idx % num_heads)
                result = {
                    'layer': layer,
                    'head': head
                }
                f_out.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    intervention_impact_attn_weights_wikidata(
        'intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/llm_judge_results.jsonl',
        # 'probe-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation/Llama-3.1-8B-Instruct/t0_fixed_is/eager/llm_judge_results.jsonl',
        'intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer6-14_alpha1.2/llm_judge_results.jsonl',
        'meta-llama/Llama-3.1-8B-Instruct',
        token_position=0,
        save_heads=True,
        save_heads_path='intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/top_heads/wikidata_test_continuation/negative-positive-steer_fixed-is_eager_layer6-14_alpha1.2/top100_heads_pos0.jsonl', )

    intervention_impact_attn_weights_celebrity(
        'intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/celebrity_test_continuation/eager/layer6-14_alpha1.2/llm_judge_results.jsonl',
        # 'probe-outputs/celebrity/celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation/Llama-3.1-8B-Instruct/t0_fixed_is/eager/llm_judge_results.jsonl',
        'intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/celebrity_test_continuation/eager/layer6-14_alpha1.2/llm_judge_results.jsonl',
        'meta-llama/Llama-3.1-8B-Instruct',
        token_position=0,
        save_heads=True,
        save_heads_path='intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/top_heads/celebrity_test_continuation/negative-positive-steer_fixed-is_eager_layer6-14_alpha1.2/top100_heads_pos0.jsonl', )