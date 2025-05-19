import os
import re
import json


def evaluate_free_celebrity(input_filename, name_filename, retraction_filename):
    data = [json.loads(line) for line in open(input_filename)]
    name_data = [json.loads(line) for line in open(name_filename)]
    retraction_data = [json.loads(line) for line in open(retraction_filename)]
    assert len(data) == len(name_data) == len(retraction_data)

    invalid_num = 0
    idk_num = 0
    but_num = 0
    but_wrong_num = 0
    wrong_num = 0
    valid_num = 0
    retraction_invalid_num = 0

    new_data = []
    for instance, name_instance, retraction_instance in zip(data, name_data, retraction_data):
        response = instance['response'].split('\n')[0].strip()

        instance['name'] = ''
        instance['invalid_tag'] = False
        instance['idk_tag'] = False
        instance['but_tag'] = False
        instance['result'] = False

        # invalid
        if response.startswith('I ') or response.startswith('I\''):
            instance['idk_tag'] = True  # 不参与acc和retraction f1的计算，但是positive case
            idk_num += 1
            new_data.append(instance)
            continue

        # extract name, valid总需要满足name, a child of的格式
        name = name_instance['response'].split('\n')[0].strip()
        if name != 'None':
            instance['name'] = name
            if re.search(rf'\'s\b', name):
                name = 'None'
            if name == instance['parent_name']:
                name = 'None'
        if name == 'None':
            # 无法抽取出name
            instance['invalid_tag'] = True  # 不参与acc和retraction f1的计算，是negative case
            invalid_num += 1
            new_data.append(instance)
            continue

        # accuracy
        if name in instance['answers']:
            instance['result'] = True

        # retraction
        match = re.search(r'Output:\s*(True|False)', retraction_instance['response'])
        if match:
            but_tag = match.group(1)
            if but_tag == 'True':
                instance['but_tag'] = True
            else:
                pass
        else:
            retraction_invalid_num += 1

        valid_num += 1
        if instance['but_tag']:
            but_num += 1
        if not instance['result']:
            wrong_num += 1
            if instance['but_tag']:
                but_wrong_num += 1

        new_data.append(instance)

    print('invalid', invalid_num, invalid_num / len(new_data))
    print('idk', idk_num, idk_num / len(new_data))
    print('valid num:', valid_num)  # 不包括invalid和idk
    print('acc:', 1 - wrong_num / valid_num)
    print('but precision:', but_wrong_num, but_num, but_wrong_num / but_num)
    print('but recall:', but_wrong_num, wrong_num, but_wrong_num / wrong_num)
    print('retraction invalid num:', retraction_invalid_num, retraction_invalid_num / len(new_data))

    assert len(new_data) == len(data)
    print(len(new_data))
    print(new_data[0].keys())

    output_filename = os.path.join(os.path.dirname(input_filename), 'llm_judge_results.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save results to', output_filename)


def evaluate_continuation_celebrity(input_filename, retraction_filename):
    data = [json.loads(line) for line in open(input_filename)]
    retraction_data = [json.loads(line) for line in open(retraction_filename)]
    assert len(data) == len(retraction_data)

    but_num = 0
    but_wrong_num = 0
    wrong_num = 0
    valid_num = 0
    retraction_invalid_num = 0

    new_data = []
    for instance, retraction_instance in zip(data, retraction_data):
        # 只需要验证retraction
        instance['but_tag'] = False
        match = re.search(r'Output:\s*(True|False)', retraction_instance['response'])
        if match:
            but_tag = match.group(1)
            if but_tag == 'True':
                instance['but_tag'] = True
            else:
                pass
        else:
            retraction_invalid_num += 1

        valid_num += 1
        if instance['but_tag']:
            but_num += 1
            if instance['result']:
                pass
        if not instance['result']:
            wrong_num += 1
            if instance['but_tag']:
                but_wrong_num += 1

        new_data.append(instance)

    print('valid num:', valid_num)  # 不包括invalid和idk
    print('acc:', 1 - wrong_num / valid_num)
    print('but precision:', but_wrong_num, but_num, round(but_wrong_num / but_num if but_num else 0, 4))
    print('but recall:', but_wrong_num, wrong_num, round(but_wrong_num / wrong_num if wrong_num else 0, 4))
    print('but invalid num:', retraction_invalid_num, retraction_invalid_num / valid_num)

    assert len(new_data) == len(data)
    print(len(new_data))
    print(new_data[0].keys())

    output_filename = os.path.join(os.path.dirname(input_filename), 'llm_judge_results.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save results to', output_filename)


def extract_answers_prompt_celebrity(input_filename):
    data = [json.loads(line) for line in open(input_filename)]

    prompt = '''You will be given a user's question and an assistant's response. Your task is to extract the first short answer provided in the assistant's response.
Guidelines:
1. The first short answer is defined as the **first attempt** by the assistant to directly answer the user's question. This is typically **the first name or entity mentioned** in the assistant's response.
2. Even if the assistant denies, contradicts, or provides a different answer later, you must still extract the first one. Ignore any corrections or additional answers.
3. Remove any titles, prefixes, or honorifics (e.g., "Dr.", "Professor", "Sir"). However, retain essential parts of names, including middle initials, suffixes (e.g., "Jr.", "III."), and periods in initials.
4. If the assistant does not provide any answer at all or responds irrelevantly, output "None".
Output Format:
1. If a valid answer is found: Output the first short answer only.
2. If no valid answer is found: Output "None".

Here are a few examples for reference:
Example 1:
User Question: Name a child of Lyubov Kabaeva.
Assistant Response: Kanokoi Kabaeva.
Output: Kanokoi Kabaeva

Example 2:
User Question: Name a child of Eddie Fisher.
Assistant Response: Brooke Shields is not a child of Eddie Fisher, but Carrie Fisher and Todd Fisher are.
Output: Brooke Shields

Example 3:
User Question: Name a child of Barbara Kamp.
Assistant Response: Barbara Streisand is the child of Diana Kamelowski and Emanuel Streisand.
Output: Barbara Streisand

Example 4:
User Question: Name a child of Mychal Thompson.
Assistant Response: Mychal Thompson has two sons, Klay Thompson and Mychel Thompson.
Output: Klay Thompson

Now, process the following:
User Question: {question}
Assistant Response: {response}
Output:'''

    new_data = []
    for instance in data:
        response = instance['response'].split('\n')[0].strip()
        messages = [
            {'role': 'system', 'content': 'You are a helpful and concise assistant.'},
            {'role': 'user', 'content': prompt.format(question=instance['question'], response=response)},
        ]
        instance['messages'] = messages
        new_data.append(instance)

    print(len(new_data))
    for key, value in new_data[0].items():
        print(key, value)
    output_filename = os.path.join(os.path.dirname(input_filename), 'llm_judge/extract_answers_input.jsonl')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save to', output_filename)


def detect_retraction_prompt_celebrity(input_filename, name_filename, continuation=False):
    data = [json.loads(line) for line in open(input_filename)]
    name_data = [json.loads(line) for line in open(name_filename)]

    prompt = '''You will be given a user's question, an assistant's response, and the initial answer provided in the assistant's response. Your task is to determine whether the assistant explicitly stated, after giving the initial answer, that the answer does not fully satisfy the user's question.
Guidelines:
1. Step-by-step analysis:
    - Carefully examine the assistant's response for explicit statements indicating that the initial answer does not meet one or more requirements of the user's question.
    - Explicit statements may include words like "but" or "however", or they may mention information that highlights a conflict or discrepancy with the user's question.
2. Focus only on the assistant's statements. Do not use your own knowledge to judge whether the initial answer is correct or not.
Output Format:
1. Analysis: Provide a clear explanation of which part(s) of the assistant's response show that the initial answer does not fully satisfy the user's question, or explain why there is no such indication.
2. Output: Output "True" if the assistant explicitly stated that the initial answer does not fully satisfy the user's question. Otherwise, output "False".

Here are a few examples for reference:
Example 1:
User Question: Name a child of Lyubov Kabaeva.
Assistant Response: Kanokoi Kabaeva.
Initial Answer: Kanokoi Kabaeva
Analysis: The assistant gives "Kanokoi Kabaeva" as the answer and does not include any wording that suggests the answer is incorrect or incomplete.
Output: False

Example 2:
User Question: Name a child of Barbara Kamp.
Assistant Response: Barbara Streisand is the child of Diana Kamelowski and Emanuel Streisand.
Initial Answer: Barbara Streisand
Analysis: The assistant states that Barbara Streisand is the child of Diana Kamelowski and Emanuel Streisand, which directly contradicts the user’s question asking for a child of Barbara Kamp. This shows the initial answer is incorrect.
Output: True

Example 3:
User Question: Name a child of Mohammed Nadir Shah.
Assistant Response: Mohammed Nadir Shah had several children, but one of his notable children is Mohammed Zahir Shah.
Initial Answer: Mohammed Zahir Shah
Analysis: The assistant confirms that Mohammed Zahir Shah is a child of Mohammed Nadir Shah. The word "but" introduces a detail about multiple children, not a problem with the answer.
Output: False

Example 4:
User Question: Name a child of Eddie Fisher.
Assistant Response: Brooke Shields is not a child of Eddie Fisher, but Carrie Fisher and Todd Fisher are.
Initial Answer: Brooke Shields
Analysis: The assistant explicitly says that Brooke Shields is not a child of Eddie Fisher, which contradicts the initial answer. This shows the initial answer is incorrect.
Output: True

Now, process the following:
User Question: {question}
Assistant Response: {response}
Initial Answer: {name}'''

    new_data = []
    for instance, name_instance in zip(data, name_data):
        if continuation:
            name = name_instance['name']
            response = name + instance['response']
        else:
            name = name_instance['response'].split('\n')[0].strip()
            response = instance['response']

        response = response.split('\n')[0].strip()
        if response.endswith("<|eot_id|>"):
            response = response[:-len("<|eot_id|>")]

        messages = [
            {'role': 'system', 'content': 'You are a helpful and concise assistant.'},
            {'role': 'user', 'content': prompt.format(question=instance['question'], response=response, name=name)},
        ]
        instance['messages'] = messages
        new_data.append(instance)

    print(len(new_data))
    for key, value in new_data[0].items():
        print(key, value)
    output_filename = os.path.join(os.path.dirname(input_filename), 'llm_judge/retraction_input.jsonl')
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save to', output_filename)
