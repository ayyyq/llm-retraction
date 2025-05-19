import os
import json
import h5py
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attn_weights(input_filename, tokenizer_path):
    data = [json.loads(line) for line in open(input_filename)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    prefix = ['<|begin_of_text|>', '<|start_header_id|>', 'system', '<|end_header_id|>', 'ĊĊ', 'Cut', 'ting', 'ĠKnowledge', 'ĠDate', ':', 'ĠDecember', 'Ġ', '202', '3', 'Ċ', 'Today', 'ĠDate', ':', 'Ġ', '26', 'ĠJul', 'Ġ', '202', '4', 'ĊĊ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', 'Ġdesigned', 'Ġto', 'Ġanswer', 'Ġquestions', '.', 'ĠAlways', 'Ġbegin', 'Ġresponses', 'Ġwith', 'Ġthe', 'Ġdirect', 'Ġanswer', '.', '<|eot_id|>', '<|start_header_id|>', 'user', '<|end_header_id|>', 'ĊĊ']
    prefix_len = len(prefix)

    plotted = []
    for index, instance in enumerate(data):
        if not instance['result'] and instance['but_tag']:
            if instance['question'] in plotted:
                continue
            plotted.append(instance['question'])

            encoding = tokenizer(instance['prompt'], add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            q_len = len(tokens)

            assert prefix == tokens[:prefix_len]
            attn_tokens = tokens[prefix_len:]

            with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
                attn_weights = f['attn_weights'][index]  # [num_layers, num_heads, seq_len]

            num_layers, num_heads, max_seq_len = attn_weights.shape
            assert q_len >= max_seq_len or attn_weights[:, :, q_len:].sum() == 0

            attn_weights = attn_weights[:, :, prefix_len:q_len]
            attn_weights = np.transpose(attn_weights, (2, 0, 1)).reshape(q_len - prefix_len, -1)

            # plot
            column_labels = []
            for l in range(num_layers):
                for h in range(num_heads):
                    if h == 0:
                        column_labels.append(f"l{l}_h0")
                    else:
                        column_labels.append("")

            plt.figure(figsize=(min(0.25 * attn_weights.shape[1], 30), min(0.35 * attn_weights.shape[0], 30)))
            sns.heatmap(attn_weights, xticklabels=column_labels, yticklabels=attn_tokens, cmap="Blues", annot=False, cbar=True)

            plt.xticks(rotation=90, fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("Attention Heads", fontsize=16)
            plt.ylabel("Tokens", fontsize=16)
            plt.title("False_True Attention Weights Heatmap", fontsize=18)
            plt.tight_layout()
            plt.show()

            pass


def plot_attentions(input_filename, tokenizer_path):
    data = [json.loads(line) for line in open(input_filename)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    prefix = ['<|begin_of_text|>', '<|start_header_id|>', 'system', '<|end_header_id|>', 'ĊĊ', 'Cut', 'ting', 'ĠKnowledge', 'ĠDate', ':', 'ĠDecember', 'Ġ', '202', '3', 'Ċ', 'Today', 'ĠDate', ':', 'Ġ', '26', 'ĠJul', 'Ġ', '202', '4', 'ĊĊ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', 'Ġdesigned', 'Ġto', 'Ġanswer', 'Ġquestions', '.', 'ĠAlways', 'Ġbegin', 'Ġresponses', 'Ġwith', 'Ġthe', 'Ġdirect', 'Ġanswer', '.', '<|eot_id|>', '<|start_header_id|>', 'user', '<|end_header_id|>', 'ĊĊ']
    prefix_len = len(prefix)

    plotted = []
    for index, instance in enumerate(data):
        if instance['result'] and not instance['but_tag']:
            if instance['question'] in plotted:
                continue
            plotted.append(instance['question'])

            encoding = tokenizer(instance['prompt'], add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            q_len = len(tokens)

            assert prefix == tokens[:prefix_len]
            attn_tokens = tokens[prefix_len:]

            with h5py.File(os.path.join(os.path.dirname(input_filename), 'attentions.h5'), 'r') as f:
                attentions = f['activations'][index]  # [num_layers, seq_len, hidden_size]

            num_layers, max_seq_len, hidden_size = attentions.shape
            assert q_len >= max_seq_len or attentions[:, q_len:, :].sum() == 0

            attentions = attentions[:, prefix_len:q_len, :]
            attn_norms = np.linalg.norm(attentions, axis=2).T  # [seq_len, num_layers]

            # plot
            column_labels = [f"l{l}" for l in range(num_layers)]

            plt.figure(figsize=(max(12, 0.4 * attn_norms.shape[1]), min(0.5 * attn_norms.shape[0], 30)))
            sns.heatmap(attn_norms, xticklabels=column_labels, yticklabels=attn_tokens, cmap="Blues", annot=False, cbar=True)

            plt.xticks(rotation=90, fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel("Layers", fontsize=16)
            plt.ylabel("Tokens", fontsize=16)
            plt.title("True_False Attention Contributions Heatmap", fontsize=18)
            plt.tight_layout()
            plt.show()

            pass


def find_offsets(s, t, offsets):
    assert t in s
    start_char = s.rfind(t)
    end_char = start_char + len(t)

    token_indices = [
        i for i, (start, end) in enumerate(offsets)
        if start < end_char and end > start_char
    ]
    return token_indices[0], token_indices[-1]


def print_top_bottom_heads(proportion_matrix, name, n=3):
    print(f"\nTop {n} heads for: {name}")
    flat = proportion_matrix.reshape(-1)  # Flatten to 1D
    top_indices = np.argsort(flat)[-n:][::-1]  # Indices of top-n
    for idx in top_indices:
        layer = idx // proportion_matrix.shape[1]
        head = idx % proportion_matrix.shape[1]
        value = proportion_matrix[layer, head]
        print(f"  Layer {layer}, Head {head}: {value:.4f}")

    print(f"\nBottom {n} heads for: {name}")
    bottom_indices = np.argsort(flat)[:n]  # Indices of bottom-n
    for idx in bottom_indices:
        layer = idx // proportion_matrix.shape[1]
        head = idx % proportion_matrix.shape[1]
        value = proportion_matrix[layer, head]
        print(f"  Layer {layer}, Head {head}: {value:.4f}")


def plot_heatmap(matrix, title):
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, annot=False, cmap='Blues', cbar=True)
    plt.title(title)
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.show()


def compare_attn_weights(input_filename, tokenizer_path, agg='max'):
    data = [json.loads(line) for line in open(input_filename)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
        num_layers, num_heads, max_seq_len = f['attn_weights'][0].shape  # [num_layers, num_heads, seq_len]

    city_count_false_total = np.zeros((num_layers, num_heads))
    city_count_false_but = np.zeros((num_layers, num_heads))
    name_count_false_total = np.zeros((num_layers, num_heads))
    name_count_false_but = np.zeros((num_layers, num_heads))

    true_false_count_total = np.zeros((num_layers, num_heads))
    true_false_count_city = np.zeros((num_layers, num_heads))
    false_false_count_total = np.zeros((num_layers, num_heads))
    false_false_count_city = np.zeros((num_layers, num_heads))
    false_true_count_total = np.zeros((num_layers, num_heads))
    false_true_count_city = np.zeros((num_layers, num_heads))

    for index, instance in enumerate(data):
        encoding = tokenizer(instance['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        q_len = len(tokens)

        offsets = encoding['offset_mapping']
        city_offsets = find_offsets(instance['prompt'], instance['city'], offsets)
        name_offsets = find_offsets(instance['prompt'], instance['name'], offsets)

        with h5py.File(os.path.join(os.path.dirname(input_filename), 'attn_weights.h5'), 'r') as f:
            attn_weights = f['attn_weights'][index]  # [num_layers, num_heads, seq_len]
        assert q_len >= max_seq_len or attn_weights[:, :, q_len:].sum() == 0

        city_attn_weights = attn_weights[:, :, city_offsets[0]:city_offsets[1]+1]
        name_attn_weights = attn_weights[:, :, name_offsets[0]:name_offsets[1]+1]

        if agg == 'max':
            city_attn = np.max(city_attn_weights, axis=2)
            name_attn = np.max(name_attn_weights, axis=2)
        elif agg == 'mean':
            city_attn = np.mean(city_attn_weights, axis=2)
            name_attn = np.mean(name_attn_weights, axis=2)
        else:
            raise NotImplementedError

        if not instance['result']:
            city_count_false_total += (city_attn > name_attn)
            name_count_false_total += (name_attn > city_attn)
            if instance['but_tag']:
                city_count_false_but += (city_attn > name_attn)
                name_count_false_but += (name_attn > city_attn)

        if instance['result'] and not instance['but_tag']:
            true_false_count_total += 1
            true_false_count_city += (city_attn > name_attn)
        elif not instance['result'] and not instance['but_tag']:
            false_false_count_total += 1
            false_false_count_city += (city_attn > name_attn)
        elif not instance['result'] and instance['but_tag']:
            false_true_count_total += 1
            false_true_count_city += (city_attn > name_attn)

    epsilon = 1e-8

    # 在false的情况下，一共有10个city_attn > name_attn，其中有2个but_tag为True
    proportion_but_given_city_gt_name = city_count_false_but / (city_count_false_total + epsilon)
    proportion_but_given_city_lt_name = name_count_false_but / (name_count_false_total + epsilon)

    proportion_city_gt_name_given_true_false = true_false_count_city / (true_false_count_total + epsilon)
    proportion_city_gt_name_given_false_false = false_false_count_city / (false_false_count_total + epsilon)
    proportion_city_gt_name_given_false_true = false_true_count_city / (false_true_count_total + epsilon)

    # print_top3_heads(proportion_but_given_city_gt_name, "P(but_tag=True | city > name & result=False)")
    # print_top3_heads(proportion_but_given_city_lt_name, "P(but_tag=True | name > city & result=False)")
    # print_top3_heads(proportion_city_gt_name_given_true_false, "P(city > name | result=True & but_tag=False)")
    # print_top3_heads(proportion_city_gt_name_given_false_false, "P(city > name | result=False & but_tag=False)")
    # print_top3_heads(proportion_city_gt_name_given_false_true, "P(city > name | result=False & but_tag=True)")

    # plot_heatmap(proportion_but_given_city_gt_name, "P(but_tag=True | city > name & result=False)")
    # plot_heatmap(proportion_but_given_city_lt_name, "P(but_tag=True | name > city & result=False)")
    # plot_heatmap(proportion_city_gt_name_given_true_false, "P(city > name | result=True & but_tag=False)")
    # plot_heatmap(proportion_city_gt_name_given_false_false, "P(city > name | result=False & but_tag=False)")
    # plot_heatmap(proportion_city_gt_name_given_false_true, "P(city > name | result=False & but_tag=True)")

    print(f'P(but_tag=True | city > name & result=False): {np.mean(proportion_but_given_city_gt_name)}')
    print(f'P(but_tag=True | name > city & result=False): {np.mean(proportion_but_given_city_lt_name)}')
    print(f'P(city > name | result=True & but_tag=False): {np.mean(proportion_city_gt_name_given_true_false)}')
    print(f'P(city > name | result=False & but_tag=False): {np.mean(proportion_city_gt_name_given_false_false)}')
    print(f'P(city > name | result=False & but_tag=True): {np.mean(proportion_city_gt_name_given_false_true)}')

    # P(but_tag=True | city > name & result=False): 0.3705096408167205
    # P(but_tag=True | name > city & result=False): 0.5985347754116651
    # P(city > name | result=True & but_tag=False): 0.37603361253649425
    # P(city > name | result=False & but_tag=False): 0.3784105705349087
    # P(city > name | result=False & but_tag=True): 0.31529349034672716


def compare_attn_contributions(input_filename, tokenizer_path, agg='max'):
    data = [json.loads(line) for line in open(input_filename)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with h5py.File(os.path.join(os.path.dirname(input_filename), 'attentions.h5'), 'r') as f:
        num_layers = f['activations'].shape[1]  # [bsz, num_layers, seq_len, hidden_size]

    # 注意：不再有 head，所有统计为 [num_layers]
    city_count_false_total = np.zeros((num_layers,))
    city_count_false_but = np.zeros((num_layers,))
    name_count_false_total = np.zeros((num_layers,))
    name_count_false_but = np.zeros((num_layers,))

    true_false_count_total = np.zeros((num_layers,))
    true_false_count_city = np.zeros((num_layers,))
    false_false_count_total = np.zeros((num_layers,))
    false_false_count_city = np.zeros((num_layers,))
    false_true_count_total = np.zeros((num_layers,))
    false_true_count_city = np.zeros((num_layers,))

    for index, instance in tqdm(enumerate(data)):
        encoding = tokenizer(instance['prompt'], add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding['offset_mapping']
        city_offsets = find_offsets(instance['prompt'], instance['city'], offsets)
        name_offsets = find_offsets(instance['prompt'], instance['name'], offsets)

        with h5py.File(os.path.join(os.path.dirname(input_filename), 'attentions.h5'), 'r') as f:
            activations = f['activations'][index]  # [num_layers, seq_len, hidden_size]

        city_attns = activations[:, city_offsets[0]:city_offsets[1]+1, :]  # [num_layers, span_len, hidden_size]
        name_attns = activations[:, name_offsets[0]:name_offsets[1]+1, :]

        city_contrib = np.linalg.norm(city_attns, axis=2)  # [num_layers, span_len]
        name_contrib = np.linalg.norm(name_attns, axis=2)

        if agg == 'max':
            city_scores = np.max(city_contrib, axis=1)  # [num_layers]
            name_scores = np.max(name_contrib, axis=1)
        elif agg == 'mean':
            city_scores = np.mean(city_contrib, axis=1)
            name_scores = np.mean(name_contrib, axis=1)
        else:
            raise NotImplementedError

        # boolean mask: city > name
        city_gt = city_scores > name_scores
        name_gt = name_scores > city_scores

        if not instance['result']:
            city_count_false_total += city_gt
            name_count_false_total += name_gt
            if instance['but_tag']:
                city_count_false_but += city_gt
                name_count_false_but += name_gt

        if instance['result'] and not instance['but_tag']:
            true_false_count_total += 1
            true_false_count_city += name_gt
        elif not instance['result'] and not instance['but_tag']:
            false_false_count_total += 1
            false_false_count_city += name_gt
        elif not instance['result'] and instance['but_tag']:
            false_true_count_total += 1
            false_true_count_city += name_gt

    epsilon = 1e-8

    proportion_but_given_city_gt_name = city_count_false_but / (city_count_false_total + epsilon)
    proportion_but_given_city_lt_name = name_count_false_but / (name_count_false_total + epsilon)

    proportion_city_gt_name_given_true_false = true_false_count_city / (true_false_count_total + epsilon)
    proportion_city_gt_name_given_false_false = false_false_count_city / (false_false_count_total + epsilon)
    proportion_city_gt_name_given_false_true = false_true_count_city / (false_true_count_total + epsilon)

    print(f'P(but_tag=True | city > name & result=False): {np.mean(proportion_but_given_city_gt_name):.4f}')
    print(f'P(but_tag=True | name > city & result=False): {np.mean(proportion_but_given_city_lt_name):.4f}')
    print(f'P(name > city | result=True & but_tag=False): {np.mean(proportion_city_gt_name_given_true_false):.4f}')
    print(f'P(name > city | result=False & but_tag=False): {np.mean(proportion_city_gt_name_given_false_false):.4f}')
    print(f'P(name > city | result=False & but_tag=True): {np.mean(proportion_city_gt_name_given_false_true):.4f}')


def intervention_impact_attn_weights(input_filename, intervention_filename, tokenizer_path, token_position=-1, agg='sum', save_heads_path=None, save_heads=False):
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

    print_top_bottom_heads(profession_decrease_avg, "Avg decrease in profession attention (interv - original)")
    print_top_bottom_heads(city_decrease_avg, "Avg decrease in city attention (interv - original)")
    print_top_bottom_heads(name_increase_avg, "Avg increase in name attention (interv - original)")

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
    # top_indices = np.argsort(flat_scores)[::-1]
    top_labels = [f"L{idx // num_heads}_H{idx % num_heads}" for idx in top_indices]
    top_values = flat_scores[top_indices]

    # plt.figure(figsize=(16, 8))
    # plt.bar(top_labels, top_values)
    # plt.xticks(rotation=90)
    # # plt.xticks([])
    # plt.ylabel("Name Increase (Δ Attn)")
    # plt.title("Attention Head Impact (Sorted)")
    # plt.tight_layout()
    # plt.show()

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

    print_top_bottom_heads(parent_decrease_avg, "Avg decrease in parent attention (interv - original)")
    print_top_bottom_heads(name_increase_avg, "Avg increase in name attention (interv - original)")

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
    # top_indices = np.argsort(flat_scores)[::-1]
    top_labels = [f"L{idx // num_heads}_H{idx % num_heads}" for idx in top_indices]
    top_values = flat_scores[top_indices]

    plt.figure(figsize=(16, 8))
    plt.bar(top_labels, top_values)
    plt.xticks(rotation=90)
    # plt.xticks([])
    plt.ylabel("Name Increase (Δ Attn)")
    plt.title("Attention Head Impact (Sorted)")
    plt.tight_layout()
    plt.show()

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


def intervention_impact_attn_contributions(input_filename, intervention_filename, tokenizer_path, token_position=-1, agg='max', save_heads_path=None, save_heads=False):
    data = [json.loads(line) for line in open(input_filename)]
    interv_data = [json.loads(line) for line in open(intervention_filename)]
    assert len(data) == len(interv_data)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with h5py.File(os.path.join(os.path.dirname(input_filename), 'attentions.h5'), 'r') as f:
        num_layers, num_tokens, max_seq_len = f['activations'][0].shape  # [num_layers, num_tokens, seq_len]

    topk_heads = None
    if not save_heads and save_heads_path is not None:
        topk_heads = [json.loads(line) for line in open(save_heads_path)][:48]

    city_decrease_sum = np.zeros((num_layers,))
    name_increase_sum = np.zeros((num_layers,))

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
        city_offsets = find_offsets(instance['prompt'], instance['city'], offsets)
        name_offsets = find_offsets(instance['prompt'], instance['name'], offsets)

        with h5py.File(os.path.join(os.path.dirname(input_filename), 'attentions.h5'), 'r') as f:
            attn_weights = f['activations'][index][:, token_position]  # [num_layers, seq_len]

        with h5py.File(os.path.join(os.path.dirname(intervention_filename), 'attentions.h5'), 'r') as f:
            interv_attn_weights = f['activations'][index][:, token_position]

        city_attn_weights = attn_weights[:, city_offsets[0]:city_offsets[1] + 1]
        name_attn_weights = attn_weights[:, name_offsets[0]:name_offsets[1] + 1]

        city_interv_attn_weights = interv_attn_weights[:, city_offsets[0]:city_offsets[1] + 1]
        name_interv_attn_weights = interv_attn_weights[:, name_offsets[0]:name_offsets[1] + 1]

        if agg == 'max':
            city_attn = np.max(city_attn_weights, axis=1)
            name_attn = np.max(name_attn_weights, axis=1)
            city_interv_attn = np.max(city_interv_attn_weights, axis=1)
            name_interv_attn = np.max(name_interv_attn_weights, axis=1)
        elif agg == 'mean':
            city_attn = np.mean(city_attn_weights, axis=1)
            name_attn = np.mean(name_attn_weights, axis=1)
            city_interv_attn = np.mean(city_interv_attn_weights, axis=1)
            name_interv_attn = np.mean(name_interv_attn_weights, axis=1)
        elif agg == 'sum':
            city_attn = np.sum(city_attn_weights, axis=1)
            name_attn = np.sum(name_attn_weights, axis=1)
            city_interv_attn = np.sum(city_interv_attn_weights, axis=1)
            name_interv_attn = np.sum(name_interv_attn_weights, axis=1)
        else:
            raise NotImplementedError

        city_decrease_sum += (city_attn - city_interv_attn)
        name_increase_sum += (name_interv_attn - name_attn)

    city_decrease_avg = city_decrease_sum / total
    name_increase_avg = name_increase_sum / total

    # print_top_bottom_heads(city_decrease_avg, "Avg decrease in city attention (interv - original)")
    # print_top_bottom_heads(name_increase_avg, "Avg increase in name attention (interv - original)")

    print(f'\nMean Δ attention (city decrease): {np.mean(city_decrease_avg):.4f}')
    print(f'Mean Δ attention (name increase): {np.mean(name_increase_avg):.4f}')

    # if topk_heads is not None:
    #     topk_city_decrease_avg = [city_decrease_avg[h['layer'], h['head']] for h in topk_heads]
    #     topk_name_increase_avg = [name_increase_avg[h['layer'], h['head']] for h in topk_heads]
    #     print(f'Mean Δ attention (city decrease) for top heads: {np.mean(topk_city_decrease_avg):.4f}')
    #     print(f'Mean Δ attention (name increase) for top heads: {np.mean(topk_name_increase_avg):.4f}')

    combined_score = name_increase_avg + city_decrease_avg
    num_layers = combined_score.shape[0]

    top_n = num_layers
    top_indices = np.argsort(combined_score)[-top_n:][::-1]
    top_labels = [f"L{idx}" for idx in top_indices]
    top_values = combined_score[top_indices]

    plt.figure(figsize=(12, 8))
    plt.bar(top_labels, top_values)
    plt.xticks(rotation=90)
    # plt.xticks([])
    plt.ylabel("Name Increase + City Decrease (Δ Attn)")
    plt.title("Attention Head Impact (Sorted)")
    plt.tight_layout()
    plt.show()

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
    # compare_attn_contributions('/home/yuqing/project/LLMDecomp/hg-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation/Llama-3.1-8B-Instruct/t0_fixed_was_born_in/eager/llm_judge_results.jsonl',
    #                 '/mnt/nfs1/yuqing/meta-llama/Llama-3.1-8B-Instruct')

    # intervention_impact_attn_weights(
    #     # '/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer8-30_alpha5.0/llm_judge_results.jsonl',
    #     '/home/yuqing/project/LLMDecomp/probe-outputs/wikidata/wikidata_continuation/OLMo-2-1124-7B-Instruct/wikidata_test_continuation/OLMo-2-1124-7B-Instruct/t0_fixed_is/eager/llm_judge_results.jsonl',
    #     '/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer8-30_alpha5.0/llm_judge_results.jsonl',
    #     '/mnt/nfs1/yuqing/allenai/OLMo-2-1124-7B-Instruct',
    #     token_position=0,
    #     save_heads=False,
    #     save_heads_path='/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/top_heads/wikidata_test_continuation/negative-positive-steer_fixed-is_eager_layer8-30_alpha5.0/top100_heads_pos0.jsonl', )

    # intervention_impact_attn_weights(
    #     '/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/positive_steer_fixed_is/wikidata_test_continuation/eager/layer10-18_alpha2.5/llm_judge_results.jsonl',
    #     '/home/yuqing/project/LLMDecomp/probe-outputs/wikidata/wikidata_continuation/Qwen2.5-7B-Instruct/wikidata_test_continuation/Qwen2.5-7B-Instruct_fp16/t0_fixed_is/eager/llm_judge_results.jsonl',
    #     # '/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/negative_steer_fixed_is/wikidata_test_continuation/eager/layer10-18_alpha2.5/llm_judge_results.jsonl',
    #     '/mnt/nfs1/yuqing/Qwen/Qwen2.5-7B-Instruct',
    #     token_position=1,
    #     save_heads=False,
    #     save_heads_path='/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/top_heads/wikidata_test_continuation/negative-positive-steer_fixed-is_eager_layer10-18_alpha2.5/top100_heads_pos1.jsonl', )

    # intervention_impact_attn_weights_celebrity(
    #     '/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/positive_steer/celebrity_test_continuation/layer8-30_alpha1.5/llm_judge_results.jsonl',
    #     '/home/yuqing/project/LLMDecomp/probe-outputs/celebrity/celebrity_continuation/OLMo-2-1124-7B-Instruct/celebrity_test_continuation/OLMo-2-1124-7B-Instruct/t0/llm_judge_results.jsonl',
    #     # '/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/negative_steer/celebrity_test_continuation/layer8-30_alpha1.5/llm_judge_results.jsonl',
    #     '/mnt/nfs1/yuqing/allenai/OLMo-2-1124-7B-Instruct',
    #     token_position=0,
    #     save_heads=False,
    #     save_heads_path='/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/top_heads/celebrity_test_continuation/negative-positive-steer_layer8-30_alpha1.5/top100_heads_pos0.jsonl', )
    #
    # intervention_impact_attn_weights_celebrity(
    #     '/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/positive_steer_fixed_is/celebrity_test_continuation/eager/layer10-18_alpha2.5/llm_judge_results.jsonl',
    #     '/home/yuqing/project/LLMDecomp/probe-outputs/celebrity/celebrity_continuation/Qwen2.5-7B-Instruct/celebrity_test_continuation/Qwen2.5-7B-Instruct_fp16/t0_fixed_is/eager/llm_judge_results.jsonl',
    #     # '/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/negative_steer_fixed_is/celebrity_test_continuation/eager/layer10-18_alpha2.5/llm_judge_results.jsonl',
    #     '/mnt/nfs1/yuqing/Qwen/Qwen2.5-7B-Instruct',
    #     token_position=0,
    #     save_heads=False,
    #     save_heads_path='/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/top_heads/celebrity_test_continuation/negative-positive-steer_fixed-is_eager_layer10-18_alpha2.5/top100_heads_pos0.jsonl', )

    # intervention_impact_attn_weights_celebrity(
    #     '/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/positive_steer_fixed_is/celebrity_test_continuation/eager/layer10-18_alpha2.5/llm_judge_results.jsonl',
    #     # '/home/yuqing/project/LLMDecomp/probe-outputs/celebrity/celebrity_continuation/Qwen2.5-7B-Instruct/celebrity_test_continuation/Qwen2.5-7B-Instruct_fp16/t0_fixed_is/eager/llm_judge_results.jsonl',
    #     '/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/negative_steer_fixed_is/celebrity_test_continuation/eager/layer10-18_alpha2.5/llm_judge_results.jsonl',
    #     '/mnt/nfs1/yuqing/Qwen/Qwen2.5-7B-Instruct',
    #     token_position=1,
    #     save_heads=True,
    #     save_heads_path='/home/yuqing/project/LLMDecomp/intervention-outputs/Qwen2.5-7B-Instruct_fp16/universal_truthfulness_train/t0/top_heads/celebrity_test_continuation/negative-positive-steer_fixed-is_eager_layer10-18_alpha2.5/top100_heads_pos1.jsonl', )


    intervention_impact_attn_weights_celebrity(
        '/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/positive_steer_fixed_is/celebrity_test_continuation/eager/layer8-30_alpha5.0/llm_judge_results.jsonl',
        # '/home/yuqing/project/LLMDecomp/probe-outputs/celebrity/celebrity_continuation/OLMo-2-1124-7B-Instruct/celebrity_test_continuation/OLMo-2-1124-7B-Instruct/t0_fixed_is/eager/llm_judge_results.jsonl',
        '/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/negative_steer_fixed_is/celebrity_test_continuation/eager/layer8-30_alpha5.0/llm_judge_results.jsonl',
        '/mnt/nfs1/yuqing/allenai/OLMo-2-1124-7B-Instruct',
        token_position=0,
        save_heads=True,
        save_heads_path='/home/yuqing/project/LLMDecomp/intervention-outputs/OLMo-2-1124-7B-Instruct/universal_truthfulness_train/t0/top_heads/celebrity_test_continuation/negative-positive-steer_fixed-is_eager_layer8-30_alpha5.0/top100_heads_pos0.jsonl', )