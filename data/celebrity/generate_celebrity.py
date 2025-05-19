import copy
import os
import sys
import json
import random
random.seed(42)


def process_negative_where_data(result_filename, where_filename):
    data = [json.loads(line) for line in open(result_filename)]
    where_data = [json.loads(line) for line in open(where_filename)]
    assert len(data) == len(where_data)

    new_data = []
    for instance, where_instance in zip(data, where_data):
        if instance['invalid_tag'] or instance['idk_tag']:
            new_data.append(instance)
            continue
        instance['parent_result'] = False
        instance['parent_response'] = where_instance['response']
        if where_instance['parent_name'] in where_instance['response']:
            instance['parent_result'] = True
        new_data.append(instance)

    assert len(data) == len(new_data)
    output_filename = os.path.join(os.path.dirname(result_filename), 'processed_llm_judge_results.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def get_negative_continuation(source_data, data_dir, seed_start, seed_end):
    for seed in range(seed_start, seed_end):
        filename = os.path.join(data_dir, f'seed{seed}/processed_llm_judge_results.jsonl')
        data = [json.loads(line) for line in open(filename)]
        assert len(data) == len(source_data)

        for index, instance in enumerate(data):
            if instance['invalid_tag'] or instance['idk_tag']:
                continue

            if instance['name'] == instance['parent_name']:
                continue

            if instance['result'] and instance['parent_result']:
                if 'positive_answers' not in source_data[index]:
                    source_data[index]['positive_answers'] = []
                source_data[index]['positive_answers'].append({'name': instance['name'],
                                                               'parent_response': instance['parent_response']})
            elif not instance['result'] and not instance['parent_result']:
                if 'negative_answers' not in source_data[index]:
                    source_data[index]['negative_answers'] = []

                source_data[index]['negative_answers'].append({'name': instance['name'],
                                                               'parent_response': instance['parent_response']})
    return source_data


def get_positive_continuation(source_data, query_filename):
    query_data = [json.loads(line) for line in open(query_filename)]

    for instance in query_data:
        index = instance['index']
        assert instance['parent_type'] == source_data[index]['parent_type']
        assert instance['parent_name'] == source_data[index]['parent_name']

        if instance['name'] == instance['parent_name']:
            continue

        if instance['parent_name'] in instance['response']:
            if 'positive_answers' not in source_data[index]:
                source_data[index]['positive_answers'] = []
            source_data[index]['positive_answers'].append({'name': instance['name'],
                                                           'parent_response': instance['response']})
    return source_data


def generate_continuation(celebrity_filename, output_filename, query_filename, data_dir, seed_start=0, seed_end=10, balanced=True, max_num=100):
    source_data = [json.loads(line) for line in open(celebrity_filename)]
    source_data = get_negative_continuation(source_data, data_dir, seed_start, seed_end)
    source_data = get_positive_continuation(source_data, query_filename)

    new_data = []

    def generate_instance(instance, item, result, index):
        question = f'Name a child of {instance["parent_name"]}.'
        messages = [
            {'role': 'system',
             'content': 'You are a helpful assistant designed to answer questions. Always begin responses with the direct answer.'},
            {'role': 'user',
             'content': question},
            {'role': 'assistant',
             'content': item['name']}
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
        new_instance['parent_response'] = item['parent_response']
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
        print('Positive data:', len(positive_data))
        print('Negative data:', len(negative_data))
        n = min(len(positive_data), len(negative_data))
        if n < 400:
            new_data = positive_data + negative_data[:1000]
        else:
            new_data = positive_data[:n] + negative_data[:n]

    print(len(new_data))
    for key, value in new_data[0].items():
        print(key, value)

    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def change_prompt(input_filename, output_filename):
    data = [json.loads(line) for line in open(input_filename)]

    new_data = []
    for instance in data:
        question = instance['question']
        messages = [
            {'role': 'user',
             'content': 'Name a child of Barack Obama. Please follow this format:\n{Name}, a child of {Parent Name}.'},
            {'role': 'assistant',
             'content': 'Malia Obama, a child of Barack Obama.'},
            {'role': 'user',
             'content': question + ' Please follow this format:\n{Name}, a child of {Parent Name}.'},
        ]

        instance['messages'] = messages
        new_data.append(instance)

    output_filename = os.path.join(os.path.dirname(input_filename), output_filename)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def test_parent_query(input_filename):
    data = [json.loads(line) for line in open(input_filename)]
    new_data = []
    for index, instance in enumerate(data):
        name = instance['name']
        question = f'Who is {name}\'s {instance["parent_type"]}?'

        messages = [
            {'role': 'system',
             'content': 'You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for.'},
            {'role': 'user',
             'content': 'Who is Elon Musk\'s mother?'},
            {'role': 'assistant',
             'content': 'Maye Musk'},
            {'role': 'user',
             'content': 'Who is Malia Obama\'s father?'},
            {'role': 'assistant',
             'content': 'Barack Obama'},
            {'role': 'user',
             'content': question}
        ]
        instance['messages'] = messages
        new_data.append(instance)

    print(len(new_data))
    output_path = os.path.join(os.path.dirname(input_filename), 'where/where_query.jsonl')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save to', output_path)


if __name__ == "__main__":
    # # 1. 处理负样本; then run negative verification questions in scripts/generate.celebrity.sh
    # for seed in range(10):
    #     test_parent_query(f'/home/yuqing/project/LLMDecomp/vllm-outputs/celebrity/celebrity_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed{seed}/llm_judge_results.jsonl')

    # # 2. 处理负样本
    # for seed in range(10):
    #     process_negative_where_data(f'/home/yuqing/project/LLMDecomp/vllm-outputs/celebrity/celebrity_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed{seed}/llm_judge_results.jsonl',
    #                                 f'/home/yuqing/project/LLMDecomp/vllm-outputs/celebrity/celebrity_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95/seed{seed}/where/where_query_output.jsonl')

    # 3. 生成model-dependent continuation dataset
    generate_continuation('celebrity_test_free.jsonl',
                          'celebrity_continuation/Llama-3.1-8B-Instruct/celebrity_test_continuation.jsonl',
                          '/home/yuqing/project/LLMDecomp/vllm-outputs/celebrity/celebrity_test_parent/Llama-3.1-8B-Instruct/t0/output.jsonl',
                          '/home/yuqing/project/LLMDecomp/vllm-outputs/celebrity/celebrity_test_free/Llama-3.1-8B-Instruct/t0.7_p0.95',
                          balanced=False)
