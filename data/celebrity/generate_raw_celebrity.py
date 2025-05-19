import copy
import os
import sys
import json
import random
random.seed(42)
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


def get_person(profession_code):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    query = f"""
        SELECT ?person ?sitelinks WHERE {{
          ?person wdt:P31 wd:Q5;  # 限制为人类
                  wdt:P106 wd:{profession_code};
                  wdt:P569 ?birthDate;
                  wikibase:sitelinks ?sitelinks.  # 获取链接数量
          FILTER (?birthDate >= "1900-01-01T00:00:00Z"^^xsd:dateTime)  # 限制出生日期
        }}
        ORDER BY DESC(?sitelinks)  # 按链接数量降序排序
        LIMIT 2000
        """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data = []
    for result in results['results']['bindings']:
        celebrity_code = result['person']['value'].split('/')[-1]
        data.append({
            'celebrity_code': celebrity_code,
        })
    # print(len(data))
    return data


def get_childs(data):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    new_data = []
    for instance in tqdm(data):
        query = f"""
            SELECT ?personLabel ?father ?mother WHERE {{
              BIND(wd:{instance['celebrity_code']} AS ?person)  # 将 Q12345 替换为目标人物的 q_code
            
              OPTIONAL {{ ?person wdt:P22 ?father. }}  # 父亲
              OPTIONAL {{ ?person wdt:P25 ?mother. }}  # 母亲
              
              SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en". 
              }}
            }}
            """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        result = results['results']['bindings'][0]

        celebrity = result['personLabel']['value']
        father_code = result['father']['value'].split('/')[-1] if 'father' in result else None
        mother_code = result['mother']['value'].split('/')[-1] if 'mother' in result else None
        if father_code or mother_code:
            instance['celebrity'] = celebrity
            instance['father'] = {'q_code': father_code, 'name': None, 'childs': []}
            instance['mother'] = {'q_code': mother_code, 'name': None, 'childs': []}
        else:
            continue

        for key, code in [('father', 'P22'), ('mother', 'P25')]:
            q_code = instance[key]['q_code']
            if q_code is None:
                continue

            query = f"""
                SELECT ?parentLabel ?child ?childLabel WHERE {{
                  BIND(wd:{q_code} AS ?parent)
                  ?child wdt:{code} ?parent.
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                """

            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            instance[key]['name'] = results['results']['bindings'][0]['parentLabel']['value']
            for result in results['results']['bindings']:
                child = result['childLabel']['value']
                instance[key]['childs'].append(child)
        new_data.append(instance)
    return new_data


def generate_raw_celebrity():
    professions = ['Q33999', 'Q82955']
    data = []
    for profession in professions:
        data += get_person(profession)
    print(len(data))
    raw_data = get_childs(data)

    print(len(raw_data))
    output_path = 'raw_celebrity.jsonl'
    with open(output_path, 'w') as f:
        for instance in raw_data:
            f.write(json.dumps(instance) + '\n')
    print('Save to', output_path)


def generate_celebrity(input_filename):
    data = [json.loads(line) for line in open(input_filename)]

    new_data = []
    for instance in data:
        for key in ['father', 'mother']:
            if instance[key]['name'] is None:
                continue
            question = f'Name a child of {instance[key]["name"]}.'
            messages = [
                {'role': 'system',
                 'content': 'You are a helpful assistant designed to answer questions. Always begin responses with the direct answer.'},
                {'role': 'user',
                 'content': question}
            ]

            new_instance = {
                'parent_type': key,
                'parent_name': instance[key]['name'],
                'answers': instance[key]['childs'],
                'question': question,
                'messages': messages
            }
            new_data.append(new_instance)

    print(len(new_data))
    return new_data


def split_train_test(data):
    random.shuffle(data)

    test_size = 800
    test_data = data[:test_size]
    train_data = data[test_size:]

    train_path = 'celebrity_train_free.jsonl'
    with open(train_path, 'w') as f:
        for instance in train_data:
            f.write(json.dumps(instance) + '\n')
    print(len(train_data))
    print(f'Save train data to {train_path}')

    test_path = 'celebrity_test_free.jsonl'
    with open(test_path, 'w') as f:
        for instance in test_data:
            f.write(json.dumps(instance) + '\n')
    print(len(test_data))
    print(f'Save test data to {test_path}')


def get_parent_query(input_filename):
    data = [json.loads(line) for line in open(input_filename)]
    new_data = []
    for index, instance in enumerate(data):
        for name in instance['answers']:
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

            new_instance = copy.deepcopy(instance)
            new_instance['question'] = question
            new_instance['messages'] = messages
            new_instance.pop('answers')
            new_instance['name'] = name
            new_instance['result'] = True
            new_instance['index'] = index
            new_data.append(new_instance)

    print(len(new_data))
    output_path = input_filename.replace('_free.jsonl', '_parent.jsonl')
    with open(output_path, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save to', output_path)


if __name__ == "__main__":
    generate_raw_celebrity()

    data = generate_celebrity('raw_celebrity.jsonl')
    split_train_test(data)

    get_parent_query('celebrity_test_free.jsonl')
