import copy
import os
import sys
import json
import random
random.seed(42)
import h5py
import numpy as np
from generate_raw_wikidata import generate_question


def filter_positive_query_data(where_filename, profession_filename):
    where_data = [json.loads(line) for line in open(where_filename)]
    profession_data = [json.loads(line) for line in open(profession_filename)]
    assert len(where_data) == len(profession_data)

    new_data = []
    question_index = []
    for where_instance, profession_instance in zip(where_data, profession_data):
        assert where_instance['index'] == profession_instance['index']
        if where_instance['city'] in where_instance['response']:
            if profession_instance['profession'].split(' ')[-1] in profession_instance['response']:
                new_data.append(where_instance)
                if where_instance['index'] not in question_index:
                    question_index.append(where_instance['index'])

    print(len(new_data))
    print('Question num:', len(question_index))
    output_filename = os.path.join(os.path.dirname(where_filename), 'filtered_output.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def get_self_where_query(input_filename, output_filename=None):
    data = [json.loads(line) for line in open(input_filename)]
    new_data = []
    for instance in data:
        name = instance['name']
        where_question = f'Where was {name} born? Please answer with the name of the city.'

        messages = [
            {"role": "system",
             "content": "You are a helpful and concise assistant. Answer questions accurately and start with the answer."},
            {"role": "user", "content": "Where was Donald Trump born? Please answer with the name of the city."},
            {"role": "assistant", "content": "New York City."},
            {"role": "user", "content": where_question}
        ]
        instance['messages'] = messages
        new_data.append(instance)

    assert len(data) == len(new_data)
    if output_filename is None:
        output_filename = os.path.join(os.path.dirname(input_filename), 'where/where_query.jsonl')
    if os.path.dirname(output_filename) and not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def process_negative_where_data(result_filename, where_filename):
    # 只保留模型在where query下回答的答案和问题中的不一样的
    data = [json.loads(line) for line in open(result_filename)]
    where_data = [json.loads(line) for line in open(where_filename)]
    assert len(data) == len(where_data)

    new_data = []
    for instance, where_instance in zip(data, where_data):
        if instance['invalid_tag'] or instance['idk_tag']:
            new_data.append(instance)
            continue
        instance['where_result'] = False
        instance['where_response'] = where_instance['response']
        if where_instance['city'] in where_instance['response']:
            instance['where_result'] = True  # 到时候扔掉
        new_data.append(instance)

    assert len(data) == len(new_data)
    output_filename = os.path.join(os.path.dirname(result_filename), 'processed_llm_judge_results.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def get_positive_continuation(source_data, query_filename):
    query_data = [json.loads(line) for line in open(query_filename)]

    for instance in query_data:
        index = instance['index']
        # 检查instance['city']是否存在于instance['response']中
        assert instance['city'] in instance['response']
        if 'positive_answers' not in source_data[index]:
            source_data[index]['positive_answers'] = []
        source_data[index]['positive_answers'].append({'name': instance['name'],
                                                       'where_response': instance['response']})

    return source_data


def get_negative_continuation(source_data, data_dir, seed_start, seed_end):
    for seed in range(seed_start, seed_end):
        filename = os.path.join(data_dir, f'seed{seed}/processed_llm_judge_results.jsonl')
        data = [json.loads(line) for line in open(filename)]
        assert len(data) == len(source_data)

        for index, instance in enumerate(data):
            if instance['invalid_tag'] or instance['idk_tag']:
                continue

            if instance['result'] and instance['where_result']:
                if 'positive_answers' not in source_data[index]:
                    source_data[index]['positive_answers'] = []
                source_data[index]['positive_answers'].append({'name': instance['name'],
                                                               'where_response': instance['where_response']})
            elif not instance['result'] and not instance['where_result']:
                if 'negative_answers' not in source_data[index]:
                    source_data[index]['negative_answers'] = []
                source_data[index]['negative_answers'].append({'name': instance['name'],
                                                               'where_response': instance['where_response']})
    return source_data


def generate_continuation(wikidata_filename, query_filename, data_dir, output_filename, seed_start=0, seed_end=2, balanced=True, max_num=100):
    source_data = [json.loads(line) for line in open(wikidata_filename)]
    source_data = get_positive_continuation(source_data, query_filename)
    source_data = get_negative_continuation(source_data, data_dir, seed_start, seed_end)

    new_data = []

    def generate_instance(instance, item, result, index):
        question = generate_question(instance['profession'], instance['city'], instance['country'])
        messages = [
            {'role': 'system',
             'content': 'You are a helpful assistant designed to answer questions. Always begin responses with the direct answer.'},
            {'role': 'user',
             'content': question},
            {'role': 'assistant',
             'content': item['name']},
        ]

        new_instance = copy.deepcopy(instance)
        new_instance['question'] = question
        new_instance['messages'] = messages
        new_instance.pop('answers')
        if 'positive_answers' in new_instance:
            new_instance.pop('positive_answers')
        if 'negative_answers' in new_instance:
            new_instance.pop('negative_answers')
        new_instance['name'] = item['name']
        new_instance['where_response'] = item['where_response']
        new_instance['result'] = result
        new_instance['index'] = index
        return new_instance

    positive_data = []
    negative_data = []
    for index, instance in enumerate(source_data):
        if 'positive_answers' in instance:
            unique_names = [name for name in set(item['name'] for item in instance['positive_answers']) if len(name.split(' ')) > 1]
            name_to_dict = {}
            for item in instance['positive_answers']:
                if item['name'] not in name_to_dict:
                    name_to_dict[item['name']] = item
            instance['positive_answers'] = [name_to_dict[name] for name in unique_names]
        if 'negative_answers' in instance:
            unique_names = [name for name in set(item['name'] for item in instance['negative_answers']) if len(name.split(' ')) > 1]
            name_to_dict = {}
            for item in instance['negative_answers']:
                if item['name'] not in name_to_dict:
                    name_to_dict[item['name']] = item
            instance['negative_answers'] = [name_to_dict[name] for name in unique_names]

        if balanced:
            if 'positive_answers' in instance and 'negative_answers' in instance:
                n = min(len(instance['positive_answers']), len(instance['negative_answers']), max_num)
                for item in instance['positive_answers'][:n]:
                    new_instance = generate_instance(instance, item, True, index)
                    new_data.append(new_instance)

                for item in instance['negative_answers'][:n]:
                    new_instance = generate_instance(instance, item, False, index)
                    new_data.append(new_instance)
        else:
            n = max_num
            if 'positive_answers' in instance:
                for item in instance['positive_answers'][:n]:
                    new_instance = generate_instance(instance, item, True, index)
                    positive_data.append(new_instance)

            if 'negative_answers' in instance:
                for item in instance['negative_answers'][:n]:
                    new_instance = generate_instance(instance, item, False, index)
                    negative_data.append(new_instance)

    if not balanced:
        random.shuffle(positive_data)
        random.shuffle(negative_data)
        n = min(len(positive_data), len(negative_data))
        new_data = positive_data[:n] + negative_data[:n]

    print(len(new_data))
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def construct_probe_dataset(input_filename, key='result', sample_size=500):
    # universal function
    data = [json.loads(line) for line in open(input_filename)]
    pos_data = []
    neg_data = []
    for index, instance in enumerate(data):
        if 'activations_index' in instance:
            assert instance['activations_index'] == index
        else:
            instance['activations_index'] = index
        if instance[key]:
            instance['label'] = True
            pos_data.append(instance)
        else:
            instance['label'] = False
            neg_data.append(instance)

    if sample_size != -1 and (sample_size < min(len(pos_data), len(neg_data)) or len(pos_data) != len(neg_data)):
        sample_size = min(len(pos_data), len(neg_data), sample_size)
        pos_data = random.sample(pos_data, sample_size)
        neg_data = random.sample(neg_data, sample_size)
        data = pos_data + neg_data
    print('pos num:', len(pos_data))
    print('neg num:', len(neg_data))

    print(len(data))
    if key == 'result':
        output_filename = 'probe_known.jsonl'
    elif key == 'but_tag':
        output_filename = 'probe_retraction.jsonl'
    else:
        raise NotImplementedError

    for key, value in data[0].items():
        print(key, value)
    output_filename = os.path.join(os.path.dirname(input_filename), output_filename)
    with open(output_filename, 'w') as f:
        for instance in data:
            f.write(json.dumps(instance) + '\n')
    print(f"Constructed dataset saved at {output_filename}")


def generate_validation(wikidata_filename, data_dir, continuation_filename, output_filename, seed_start=0, seed_end=5):
    source_data = [json.loads(line) for line in open(wikidata_filename)]
    source_data = get_negative_continuation(source_data, data_dir, seed_start, seed_end)
    continuation_data = [json.loads(line) for line in open(continuation_filename)]
    continuation_data = [(instance['index'], instance['name']) for instance in continuation_data]
    print('continuation data num:', len(continuation_data))

    def generate_instance(instance, item, result, index):
        question = generate_question(instance['profession'], instance['city'], instance['country'])
        messages = [
            {'role': 'system',
             'content': 'You are a helpful assistant designed to answer questions. Always begin responses with the direct answer.'},
            {'role': 'user',
             'content': question},
            {'role': 'assistant',
             'content': item['name']},
        ]

        new_instance = copy.deepcopy(instance)
        new_instance['question'] = question
        new_instance['messages'] = messages
        new_instance.pop('answers')
        if 'positive_answers' in new_instance:
            new_instance.pop('positive_answers')
        if 'negative_answers' in new_instance:
            new_instance.pop('negative_answers')
        new_instance['name'] = item['name']
        new_instance['where_response'] = item['where_response']
        new_instance['result'] = result
        new_instance['index'] = index
        return new_instance

    new_data = []
    for index, instance in enumerate(source_data):
        if 'negative_answers' in instance:
            unique_names = [name for name in set(item['name'] for item in instance['negative_answers']) if len(name.split(' ')) > 1]
            name_to_dict = {}
            for item in instance['negative_answers']:
                if item['name'] not in name_to_dict:
                    name_to_dict[item['name']] = item
            instance['negative_answers'] = [name_to_dict[name] for name in unique_names]

            for item in instance['negative_answers']:
                if (index, item['name']) in continuation_data:
                    continue

                new_instance = generate_instance(instance, item, False, index)
                new_data.append(new_instance)

    new_data = random.sample(new_data, min(len(new_data), 50))
    print(len(new_data))
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


if __name__ == "__main__":
    # 1. 处理正样本
    # filter_positive_query_data('vllm-outputs/wikidata/wikidata_test_where/Llama-3.1-8B-Instruct/t0/output.jsonl',
    #                            'vllm-outputs/wikidata/wikidata_test_profession/Llama-3.1-8B-Instruct/t0/output.jsonl')

    # 2. 处理负样本
    # for seed in range(5):
    #     # 2.1 生成verification questions; then run negative verification questions in scripts/generate.celebrity.sh
    #     # get_self_where_query(f'vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed{seed}/llm_judge_results.jsonl')
    #
    #     # 2.2 处理负样本
    #     process_negative_where_data(f'vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed{seed}/llm_judge_results.jsonl',
    #                                 f'vllm-outputs/wikidata/wikidata_train_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed{seed}/where/where_query_output.jsonl')

    # 3. 生成model-specific continuation
    generate_continuation('wikidata_test_free.jsonl',
                          'vllm-outputs/wikidata/wikidata_test_where/Llama-3.1-8B-Instruct/t0/filtered_output.jsonl',
                          'vllm-outputs/wikidata/wikidata_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95',
                          'data_collection/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_test_continuation.jsonl',
                          seed_start=0, seed_end=5,
                          balanced=False,
                          max_num=100)

    # # 4. Optional: 生成probe dataset for Wikidata Retraction Steering Direction
    # construct_probe_dataset(
    #     f'probe-outputs/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_train_continuation/Llama-3.1-8B-Instruct/t0/llm_judge_results.jsonl',
    #     key='but_tag',
    #     sample_size=-1)

    # model_name = 'Llama-3.1-8B-Instruct'
    # generate_validation('wikidata_train_free.jsonl',
    #                       f'vllm-outputs/wikidata/wikidata_train_free/{model_name}/t0.7_p0.95',
    #                       f'data_collection/wikidata/wikidata_continuation/{model_name}/wikidata_train_continuation.jsonl',
    #                         f'data_collection/wikidata/wikidata_continuation/{model_name}/wikidata_val_continuation.jsonl')
