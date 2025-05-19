import os
import json


def read_results(input_filename):
    data = [json.loads(line) for line in open(input_filename)]

    invalid_num = 0
    idk_num = 0
    valid_num = 0
    but_num = 0
    wrong_num = 0
    but_wrong_num = 0

    s_num = {'total': 0, 'but_tag': 0, 'not_but_tag': 0}  # \'s

    for instance in data:
        if 'invalid_tag' in instance and instance['invalid_tag']:
            invalid_num += 1
            continue
        if 'idk_tag' in instance and instance['idk_tag']:
            idk_num += 1
            continue

        valid_num += 1
        if instance['but_tag']:
            but_num += 1
            if instance['result']:
                pass

        if not instance['result']:
            wrong_num += 1
            if instance['but_tag']:
                but_wrong_num += 1

        if instance['response'].startswith('\'s'):
            s_num['total'] += 1
            if instance['but_tag']:
                s_num['but_tag'] += 1
            else:
                s_num['not_but_tag'] += 1

    print(input_filename)
    print(len(data))
    print('invalid', invalid_num, invalid_num / len(data))
    print('idk', idk_num, idk_num / len(data))
    print('valid num:', valid_num)  # 不包括invalid和idk
    assert valid_num == len(data)
    print('acc:', 1 - wrong_num / valid_num)
    print('but precision:', but_wrong_num, but_num, round(but_wrong_num / but_num if but_num != 0 else 0, 4))
    print('but recall:', but_wrong_num, wrong_num, round(but_wrong_num / wrong_num, 4))
    print('but rate:', but_num, valid_num, round(but_num / valid_num, 4))

    print(s_num)


def evaluate_end(filename, save=False):
    data = [json.loads(line) for line in open(filename)]

    total_groups = {'True_True': 0, 'True_False': 0, 'False_True': 0, 'False_False': 0}
    end_groups = {'True_True': 0, 'True_False': 0, 'False_True': 0, 'False_False': 0}
    dot_groups = {'True_True': 0, 'True_False': 0, 'False_True': 0, 'False_False': 0}
    eos_groups = {'True_True': 0, 'True_False': 0, 'False_True': 0, 'False_False': 0}
    new_data = []
    for instance in data:
        instance['end_tag'] = False

        if 'result' not in instance:
            instance['result'] = instance['label']
        if 'but_tag' not in instance:
            instance['but_tag'] = False

        if instance['result'] and instance['but_tag']:
            total_groups['True_True'] += 1
        elif instance['result'] and not instance['but_tag']:
            total_groups['True_False'] += 1
        elif not instance['result'] and instance['but_tag']:
            total_groups['False_True'] += 1
        else:
            total_groups['False_False'] += 1

        response = instance['response'].split('\n')[0].strip()
        if response == '.' or response == '' or response == '<|eot_id|>' or response == '<|im_end|>':
            instance['end_tag'] = True

            if instance['result'] and instance['but_tag']:
                end_groups['True_True'] += 1
                if response == '.':
                    dot_groups['True_True'] += 1
                else:
                    eos_groups['True_True'] += 1
            elif instance['result'] and not instance['but_tag']:
                end_groups['True_False'] += 1
                if response == '.':
                    dot_groups['True_False'] += 1
                else:
                    eos_groups['True_False'] += 1
            elif not instance['result'] and instance['but_tag']:
                end_groups['False_True'] += 1
                if response == '.':
                    dot_groups['False_True'] += 1
                else:
                    eos_groups['False_True'] += 1
            else:
                end_groups['False_False'] += 1
                if response == '.':
                    dot_groups['False_False'] += 1
                else:
                    eos_groups['False_False'] += 1
        elif len(response.split(' ')) <= 1:
            pass
        else:
            if instance['result'] and not instance['but_tag']:
                pass

        instance.pop('result')
        instance.pop('but_tag')
        new_data.append(instance)

    for key in end_groups:
        print(f"{key}: {end_groups[key]} / {total_groups[key]} = {end_groups[key] / total_groups[key] if total_groups[key] != 0 else 0}")
        print(f"{key} dot: {dot_groups[key]} / {total_groups[key]} = {dot_groups[key] / total_groups[key] if total_groups[key] != 0 else 0}")
        print(f"{key} eos: {eos_groups[key]} / {total_groups[key]} = {eos_groups[key] / total_groups[key] if total_groups[key] != 0 else 0}")
    print(f'Non-retracted samples stop generating: {round((end_groups["True_False"] + end_groups["False_False"]) / (total_groups["True_False"] + total_groups["False_False"]), 4)}')
    print(f'Stop Rate: {round((end_groups["True_True"] + end_groups["True_False"] + end_groups["False_True"] + end_groups["False_False"]) / len(data), 4)}')

    assert len(data) == len(new_data)
    if save:
        output_filename = os.path.join(os.path.dirname(filename), 'output_end.jsonl')
        # assert not os.path.exists(output_filename)
        with open(output_filename, 'w') as f:
            for instance in new_data:
                f.write(json.dumps(instance) + '\n')
        print('Save results to', output_filename)


if __name__ == '__main__':
    evaluate_end('intervention-outputs/Llama-3.1-8B-Instruct/universal_truthfulness_train/t0/negative_steer/wikidata_test_continuation/layer6-14_alpha1.2/llm_judge_results.jsonl')