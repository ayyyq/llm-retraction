import os
import json
import random
random.seed(42)


def read_dataset(input_filename, tag, sample_size=200):
    data = json.loads(open(input_filename).read())
    random.shuffle(data)

    pos_data = []
    neg_data = []
    for instance in data:
        answer = instance['answer']
        if answer.endswith('.'):
            answer = answer[:-1]
        messages = [
            {'role': 'user',
             'content': instance['question']},
            {'role': 'assistant',
             'content': answer}
        ]
        instance['messages'] = messages
        instance['tag'] = tag

        if instance['label'] == 1:
            pos_data.append(instance)
        else:
            neg_data.append(instance)

    print(f"pos_num: {len(pos_data)}, neg_num: {len(neg_data)}")
    if sample_size != -1:
        sample_size = min(sample_size, len(pos_data), len(neg_data))
        new_data = pos_data[:sample_size] + neg_data[:sample_size]
    else:
        new_data = pos_data + neg_data
    print(len(new_data))

    return new_data


if __name__ == '__main__':
    for split in ['train', 'vali']:
        data = []
        for tag in ['nq_re', 'triva_qa_re', 'sciq']:
            input_filename = f'{tag}/{tag}_probe_{split}.json'
            data += read_dataset(input_filename, tag, sample_size=-1 if split == 'train' else 400)
        for index in range(len(data)):
            data[index]['activations_index'] = index
        print(len(data))
        output_filename = f'truthfulness_test.jsonl' if split == 'vali' else f'truthfulness_{split}.jsonl'
        with open(output_filename, 'w') as f:
            for instance in data:
                f.write(json.dumps(instance) + '\n')
        print('Sampled training data saved to', output_filename)
