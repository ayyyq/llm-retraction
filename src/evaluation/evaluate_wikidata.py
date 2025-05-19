import os
import json
import re
import sys
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


def process_free_response(data, name_data, retraction_data):
    invalid_num = 0
    idk_num = 0
    but_num = 0
    retraction_invalid_num = 0
    assert len(data) == len(name_data) == len(retraction_data)

    new_data = []
    for instance, name_instance, retraction_instance in zip(data, name_data, retraction_data):
        response = instance['response'].split('\n')[0].strip()

        instance['name'] = ''
        instance['invalid_tag'] = False
        instance['idk_tag'] = False
        instance['but_tag'] = False

        # invalid
        if response.startswith('I ') or response.startswith('I\''):  # 这里没有计算非开头的I couldn't是因为可以作为一种给定答案后的retraction
            instance['idk_tag'] = True  # 不参与acc和retraction f1的计算，但是positive case
            idk_num += 1
            new_data.append(instance)
            continue

        # extract name
        name = name_instance['response'].split('\n')[0].strip()
        if name != 'None':
            instance['name'] = name
            if re.search(rf'\'s\b', name):
                name = 'None'
        if name == 'None':
            # 无法抽取出name
            instance['invalid_tag'] = True  # 不参与acc和retraction f1的计算，是negative case
            invalid_num += 1
            new_data.append(instance)
            continue

        # retraction
        # 验证birthplace是否retract
        match = re.search(r'Output:\s*(True|False)', retraction_instance['response'])
        if match:
            but_tag = match.group(1)
            if but_tag == 'True':
                instance['but_tag'] = True
                but_num += 1
            else:
                pass
        else:
            retraction_invalid_num += 1

        new_data.append(instance)

    assert len(new_data) == len(data)
    print('invalid', invalid_num, invalid_num / len(new_data))
    print('idk', idk_num, idk_num / len(new_data))
    print('but num:', but_num)
    print('retraction invalid num:', retraction_invalid_num, retraction_invalid_num / len(new_data))
    return new_data


def search_wikidata(data):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
    fake_num = 0

    def query1(escaped_name):
        query = f"""
            SELECT ?person ?actualCity ?actualCityLabel WHERE {{
              ?person wdt:P31 wd:Q5;
                     rdfs:label ?label;
                     wdt:P19 ?actualCity;
                     wdt:P106 wd:{instance['profession_code']}.
              VALUES ?label {{
                  "{escaped_name}"@en
                  "{escaped_name}"@de
                  "{escaped_name}"@fr
                  "{escaped_name}"@es
                  "{escaped_name}"@it
                  "{escaped_name}"@pt
                  "{escaped_name}"@nl
                  "{escaped_name}"@sv
                  "{escaped_name}"@da
                  "{escaped_name}"@no
                  "{escaped_name}"@pl
                  "{escaped_name}"@cs
                  "{escaped_name}"@hu
                  "{escaped_name}"@fi
                  "{escaped_name}"@bg
                  "{escaped_name}"@sr
                  "{escaped_name}"@hr
                  "{escaped_name}"@ro
                  "{escaped_name}"@el
                  "{escaped_name}"@zh
                  "{escaped_name}"@ja
                  "{escaped_name}"@ko
                  "{escaped_name}"@hi
                  "{escaped_name}"@bn
                  "{escaped_name}"@th
                  "{escaped_name}"@vi
                  "{escaped_name}"@id
                  "{escaped_name}"@ms
                  "{escaped_name}"@ar
                  "{escaped_name}"@fa
                  "{escaped_name}"@tr
                  "{escaped_name}"@he
                  "{escaped_name}"@ru
                  "{escaped_name}"@uk
                  "{escaped_name}"@sw
                  "{escaped_name}"@af
                  "{escaped_name}"@ha
                  "{escaped_name}"@yo
                  "{escaped_name}"@tl
                  "{escaped_name}"@ur
                  "{escaped_name}"@bs
                  "{escaped_name}"@sl
                  "{escaped_name}"@sk
                  "{escaped_name}"@lv
                  "{escaped_name}"@lt
                  "{escaped_name}"@et
                  "{escaped_name}"
                }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            LIMIT 10
        """

        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return results

        except Exception as e:
            print(f"Error querying for {instance['name']}: {e}")

        return None

    def query2(escaped_name):
        query = f"""
                    SELECT ?person ?actualCity ?actualCityLabel WHERE {{
                      ?person wdt:P31 wd:Q5;
                             rdfs:label ?label;
                             wdt:P19 ?actualCity;
                             wdt:P106 wd:{instance['profession_code']}.
                      FILTER(STR(?label) = "{escaped_name}")
                      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                    }}
                LIMIT 1
                """

        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return results

        except Exception as e:
            print(f"Error querying for {instance['name']}: {e}")

        return None

    new_data = []
    for index, instance in tqdm(enumerate(data)):
        instance['result'] = False
        if instance['name'] in instance['answers']:
            instance['result'] = True
            new_data.append(instance)
            continue

        escaped_name = instance['name'].encode('unicode_escape').decode('ascii').replace('\\', '\\\\').replace('"', '\\"')

        results = query1(escaped_name)
        # if results is None or not results['results']['bindings']:
        #     results = query2(escaped_name)

        if results is None or len(results['results']['bindings']) == 0:
            fake_num += 1
        else:
            for result in results['results']['bindings']:
                actual_city_code = result['actualCity']['value'].split('/')[-1]
                if actual_city_code == instance['city_code']:
                    instance['result'] = True
                    break
            if not instance['result']:
                actual_city = results['results']['bindings'][0]['actualCityLabel']['value']
                instance['actual_city'] = actual_city

        new_data.append(instance)

    print('fake num:', fake_num, fake_num / len(data))
    return new_data


def evaluate_free_wikidata(input_filename, name_filename, retraction_filename):
    # 给定问题，模型自由生成response
    data = [json.loads(line) for line in open(input_filename)]
    name_data = [json.loads(line) for line in open(name_filename)]
    retraction_data = [json.loads(line) for line in open(retraction_filename)]
    new_data = process_free_response(data, name_data, retraction_data)
    if 'result' not in new_data[0]:
        new_data = search_wikidata(new_data)

    valid_num = 0
    but_wrong_num = 0
    but_num = 0
    wrong_num = 0

    for instance in new_data:
        if instance['invalid_tag'] or instance['idk_tag']:
            continue

        valid_num += 1

        if instance['but_tag']:
            but_num += 1

        if not instance['result']:
            wrong_num += 1
            if instance['but_tag']:
                but_wrong_num += 1

    print('valid num:', valid_num)  # 不包括invalid和idk
    print('acc:', 1 - wrong_num / valid_num)
    print('but precision:', but_wrong_num, but_num, but_wrong_num / but_num)
    print('but recall:', but_wrong_num, wrong_num, but_wrong_num / wrong_num)

    assert len(new_data) == len(data)
    print(len(new_data))
    print(new_data[0].keys())

    output_filename = os.path.join(os.path.dirname(input_filename), 'llm_judge_results.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save results to', output_filename)


def evaluate_continuation_wikidata(input_filename, retraction_filename):
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
        # 只需要验证but_tag
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


def extract_answers_prompt_wikidata(input_filename):
    data = [json.loads(line) for line in open(input_filename)]

    prompt = '''You will be given a user's question and an assistant's response. Your task is to extract the first short answer provided in the assistant's response.
Guidelines:
1. The first short answer is defined as the **first attempt** by the assistant to directly answer the user's question.
2. Even if the assistant denies, contradicts, or provides a different answer later, you must still extract the first one. Ignore any corrections or additional answers.
3. Remove any titles, prefixes, or honorifics (e.g., "Dr.", "Professor", "Sir"). However, retain essential parts of names, including middle initials, suffixes (e.g., "Jr.", "III."), and periods in initials.
4. If the assistant does not provide any answer at all or responds irrelevantly, output "None".
Output Format:
1. If a valid answer is found: Output the first short answer only.
2. If no valid answer is found: Output "None".

Here are a few examples for reference:
Example 1:
User Question: Name an actor who was born in Sheffield, United Kingdom.
Assistant Response: Michael Palin.
Output: Michael Palin

Example 2:
User Question: Name a coach who was born in Chelyabinsk, Russia.
Assistant Response: Vitaly Mutko is not a coach, he is a politician.
Output: Vitaly Mutko

Example 3:
User Question: Name a sociologist who was born in Mexico City, Mexico.
Assistant Response: Pierre Bourdieu was not born in Mexico City, but a notable sociologist born in Mexico City is  Guillermo Bonfil Batalla.
Output: Pierre Bourdieu

Example 4:
User Question: Name an association football manager who was born in Sydney, Australia.
Assistant Response: Guus Hiddink, born in Melbourne, Australia.
Output: Guus Hiddink

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


def detect_retraction_prompt_wikidata(input_filename, name_filename, continuation=False):
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
User Question: Name an actor who was born in Sheffield, United Kingdom.
Assistant Response: Michael Palin.
Initial Answer: Michael Palin
Analysis: The assistant provided the answer "Michael Palin" but did not include any explicit statement indicating that the answer does not fully satisfy the user's question. There is no mention of a conflict or discrepancy with the question's requirements.
Output: False

Example 2:
User Question: Name an association football manager who was born in Sydney, Australia.
Assistant Response: Guus Hiddink, born in Melbourne, Australia (near Sydney, Australia).
Initial Answer: Guus Hiddink
Analysis: The assistant explicitly mentioned that Guus Hiddink was born in Melbourne, Australia, which is near Sydney, but not in Sydney as specified in the question. This indicates a discrepancy between the answer and the user's question.
Output: True

Example 3:
User Question: Name a pastor who was born in Magdeburg, Germany.
Assistant Response: Martin Luther was born in Eisleben, but lived in Magdeburg, Germany.
Initial Answer: Martin Luther
Analysis: The assistant explicitly stated that Martin Luther was born in Eisleben, not Magdeburg, as specified in the question. Although the assistant mentioned that he also lived in Magdeburg, this does not fulfill the birth requirement in the user's question, indicating a conflict.
Output: True

Example 4:
User Question: Name an engineer who was born in Rotterdam, Netherlands.
Assistant Response: Antonie van Leeuwenhoek was born in Rotterdam, Netherlands, but a more famous one is Michael Abramoff.
Initial Answer: Antonie van Leeuwenhoek
Analysis: The assistant clearly stated that Antonie van Leeuwenhoek was born in Rotterdam, Netherlands, which fully satisfies the user's question. The mention of a more famous person is additional information but does not indicate any issue with the initial answer.
Output: False

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
