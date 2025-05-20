import copy
import os
import json


def get_retraction_sft_data(filename):
    data = [json.loads(line) for line in open(filename)]  # wikidata_continuation

    pos, neg = 0, 0
    new_data = []
    for index, instance in enumerate(data):
        if 'result' not in instance:
            instance['result'] = instance['label']
        if 'name' not in instance:
            instance['name'] = instance['messages'][-1]['content']
        if instance['result']:
            # positive
            output = instance['name'] + ' is the correct answer.'
            pos += 1
        else:
            # negative
            output = instance['name'] + ' is not the correct answer.'
            neg += 1

        new_instance = copy.deepcopy(instance)
        new_instance['index'] = index
        if instance['messages'][0]['role'] == 'system':
            new_instance['system'] = new_instance['messages'][0]['content']
            new_instance['instruction'] = new_instance['messages'][1]['content']
        else:
            new_instance['instruction'] = new_instance['messages'][0]['content']
        new_instance['output'] = output
        new_data.append(new_instance)

    print('pos:', pos)
    print('neg:', neg)
    print(len(new_data))
    for key, value in new_data[0].items():
        print(key, value)
    output_filename = os.path.join(os.path.dirname(filename), 'retraction_sft', os.path.basename(filename).split('.')[0] + '.json')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f'Saved output to {output_filename}')


if __name__ == '__main__':
    get_retraction_sft_data('data/wikidata/wikidata_continuation/Llama-3.1-8B-Instruct/wikidata_train_continuation.jsonl')

