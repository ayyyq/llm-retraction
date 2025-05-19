import os
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(src_path)
sys.path.append(src_path)

import json
import collections

import matplotlib.pyplot as plt


def plot_probe_scores(data_dir, num_layers=32):
    group_score = {'True_True': [], 'True_False': [], 'False_True': [], 'False_False': []}

    for layer in range(num_layers):
        data_filename = os.path.join(data_dir, f'l{layer}_predictions.jsonl')

        with open(data_filename) as f:
            data = [json.loads(line) for line in f]

        _group_score = {'True_True': [], 'True_False': [], 'False_True': [], 'False_False': []}

        for instance in data:
            if instance['result'] and instance['but_tag']:
                _group_score['True_True'].append(instance['probe_pred_score'])
            elif instance['result'] and not instance['but_tag']:
                _group_score['True_False'].append(instance['probe_pred_score'])
            elif not instance['result'] and instance['but_tag']:
                _group_score['False_True'].append(instance['probe_pred_score'])
            elif not instance['result'] and not instance['but_tag']:
                _group_score['False_False'].append(instance['probe_pred_score'])
            else:
                raise ValueError("Unknown group")

        def avg(x):
            return sum(x) / len(x) if x else 0.0

        for key in _group_score:
            group_score[key].append(avg(_group_score[key]))

        print(f'Layer {layer}: '
              f'True_True={len(_group_score["True_True"])}, '
              f'True_False={len(_group_score["True_False"])}, '
              f'False_True={len(_group_score["False_True"])}, '
              f'False_False={len(_group_score["False_False"])}')

    # 绘图
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
    })
    layers = list(range(num_layers))

    CORRECT_COLOR = "#0e7177"
    WRONG_COLOR = "#d6908f"
    plot_config = {
        'True_False': {'color': CORRECT_COLOR, 'linestyle': '-', 'marker': 'o', 'label': 'CN'},  # soft green, solid
        'True_True': {'color': CORRECT_COLOR, 'linestyle': '--', 'marker': 'x', 'label': 'CR'},  # soft green, dashed
        'False_False': {'color': WRONG_COLOR, 'linestyle': '-', 'marker': 'o', 'label': 'WN'},  # soft red, solid
        'False_True': {'color': WRONG_COLOR, 'linestyle': '--', 'marker': 'x', 'label': 'WR'},  # soft red, dashed
    }

    for key, cfg in plot_config.items():
        if cfg['linestyle'] == '--':
            plt.plot(
                layers, group_score[key],
                marker=cfg['marker'], markersize=9, markeredgewidth=1.5, linewidth=2.8,
                linestyle=cfg['linestyle'], color=cfg['color'], label=cfg['label']
            )
        else:
            plt.plot(
                layers, group_score[key],
                marker=cfg['marker'], markersize=8, linewidth=2.8,
                linestyle=cfg['linestyle'], color=cfg['color'], label=cfg['label']
            )

    plt.xlabel('Layer')
    plt.ylabel('Avg. Probe Score')

    plt.xticks(list(range(0, num_layers, 4)))
    plt.xlim(0, num_layers - 1)  # 去掉左右空白
    plt.ylim(0, 1)  # 去掉上下空白

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left", frameon=True, framealpha=0.4, edgecolor='black')
    plt.tight_layout()

    if 'llama' in data_dir.lower():
        model_name = 'llama'
    elif 'qwen' in data_dir.lower():
        model_name = 'qwen'
    elif 'olmo' in data_dir.lower():
        model_name = 'olmo'
    else:
        raise ValueError("Unknown model name in data_dir")

    if 'wikidata' in data_dir.lower():
        dataset_name = 'wikidata'
    elif 'celebrity' in data_dir.lower():
        dataset_name = 'celebrity'
    else:
        raise ValueError("Unknown dataset name in data_dir")
    # plt.show()
    plt.savefig(os.path.join(data_dir, f'probe_{model_name}_{dataset_name}.pdf'), dpi=300, bbox_inches='tight')


def plot_sft_probe_scores(sft_data_dir, data_dir, num_layers=32):
    baseline_wrong_scores, sft_wrong_scores = [], []
    baseline_correct_scores, sft_correct_scores = [], []

    for layer in range(num_layers):
        sft_filename = os.path.join(sft_data_dir, f'l{layer}_predictions.jsonl')
        data_filename = os.path.join(data_dir, f'l{layer}_predictions.jsonl')

        with open(sft_filename) as f1, open(data_filename) as f2:
            sft_data = [json.loads(line) for line in f1]
            data = [json.loads(line) for line in f2]

        assert len(sft_data) == len(data)

        sft_wrong, baseline_wrong = [], []
        sft_correct, baseline_correct = [], []

        for sft_instance, instance in zip(sft_data, data):
            assert sft_instance['question'] == instance['question']
            assert sft_instance['name'] == instance['name']

            if not sft_instance['result']:
                sft_wrong.append(sft_instance['probe_pred_score'])
                baseline_wrong.append(instance['probe_pred_score'])
            elif sft_instance['result']:
                sft_correct.append(sft_instance['probe_pred_score'])
                baseline_correct.append(instance['probe_pred_score'])

        def avg(x): return sum(x) / len(x) if x else 0.0

        sft_wrong_scores.append(avg(sft_wrong))
        baseline_wrong_scores.append(avg(baseline_wrong))
        sft_correct_scores.append(avg(sft_correct))
        baseline_correct_scores.append(avg(baseline_correct))

        print(f'Layer {layer}: wrong={len(sft_wrong)}, correct={len(sft_correct)}')

    # 绘图
    plt.figure(figsize=(10, 7))
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
    })
    layers = list(range(num_layers))

    CORRECT_COLOR = "#0e7177"
    WRONG_COLOR = "#d6908f"

    plt.plot(layers, baseline_correct_scores, marker='x', markersize=9, markeredgewidth=1.5, linestyle='--', linewidth=2.8,
             color=CORRECT_COLOR, label='Base-C')
    plt.plot(layers, baseline_wrong_scores, marker='x', markersize=9, markeredgewidth=1.5, linestyle='--', linewidth=2.8,
             color=WRONG_COLOR, label='Base-W')
    plt.plot(layers, sft_correct_scores, marker='o', markersize=8, linestyle='-', linewidth=2.8,
             color=CORRECT_COLOR, label='SFT-C')
    plt.plot(layers, sft_wrong_scores, marker='o', markersize=8, linestyle='-', linewidth=2.8,
             color=WRONG_COLOR, label='SFT-W')

    plt.xlabel('Layer')
    plt.ylabel('Avg. Probe Score')
    plt.xticks(list(range(0, num_layers, 4)))
    plt.xlim(0, num_layers - 1)

    all_scores = baseline_wrong_scores + sft_wrong_scores + baseline_correct_scores + sft_correct_scores
    y_min, y_max = min(all_scores) - 0.05, max(all_scores) + 0.05
    plt.ylim(max(0, y_min), min(1.0, y_max))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="upper left", frameon=True, framealpha=0.4, edgecolor='black')
    plt.tight_layout()
    if 'llama' in data_dir.lower():
        model_name = 'llama'
    elif 'qwen' in data_dir.lower():
        model_name = 'qwen'
    elif 'olmo' in data_dir.lower():
        model_name = 'olmo'
    else:
        raise ValueError("Unknown model name in data_dir")

    if 'wikidata' in data_dir.lower():
        dataset_name = 'wikidata'
    elif 'celebrity' in data_dir.lower():
        dataset_name = 'celebrity'
    else:
        raise ValueError("Unknown dataset name in data_dir")

    # plt.show()
    plt.savefig(os.path.join(sft_data_dir, f'probe_sft_{model_name}_{dataset_name}.pdf'), dpi=300, bbox_inches='tight')
