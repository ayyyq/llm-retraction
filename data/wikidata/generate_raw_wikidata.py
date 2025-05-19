import copy
import os
import sys
import json
import random

random.seed(42)
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


def generate_question(profession, city, country):
    vowels = ['a', 'e', 'i', 'o', 'u']
    article = 'an' if profession[0].lower() in vowels else 'a'
    return f"Name {article} {profession} who was born in {city}, {country}."


def get_professions_with_high_counts():
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    query = f"""
    SELECT ?profession (COUNT(?person) AS ?numPeople) WHERE {{
      ?person wdt:P106 ?profession.
      ?profession wdt:P31 wd:Q28640.
    }}
    GROUP BY ?profession
    HAVING (COUNT(?person) > 1000)
    ORDER BY DESC(?numPeople)
    LIMIT 100"""

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    professions = []
    for result in results['results']['bindings']:
        profession_code = result['profession']['value'].split('/')[-1]
        professions.append({
            'profession_code': profession_code,
        })
    return professions


def get_profession_labels(professions):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    profession_values = ' '.join([f"wd:{profession['profession_code']}" for profession in professions])
    query = f"""
    SELECT ?profession ?professionLabel WHERE {{
    VALUES ?profession {{ {profession_values} }}  # 第一阶段返回的 Q 号
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} # 获取职业名称
    }}"""

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    profession_list = []
    for result in results['results']['bindings']:
        profession_code = result['profession']['value'].split('/')[-1]
        profession = result['professionLabel']['value']
        profession_list.append({
            'profession_code': profession_code,
            'profession': profession,
        })
    return profession_list


def get_cities_with_high_counts():
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    query = f"""
    SELECT ?city (COUNT(?person) AS ?numPeople) WHERE {{
      ?person wdt:P19 ?city.
      ?city wdt:P31 wd:Q515.
      FILTER NOT EXISTS {{ ?city wdt:P576 ?dissolvedDate. }}  # 过滤已消失城市
    }}
    GROUP BY ?city
    HAVING (COUNT(?person) > 200)
    ORDER BY DESC(?numPeople)
    LIMIT 200"""

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    cities = []
    for result in results['results']['bindings']:
        city_code = result['city']['value'].split('/')[-1]
        cities.append({
            'city_code': city_code,
        })
    return cities


def get_city_labels(cities):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    city_values = ' '.join([f"wd:{city['city_code']}" for city in cities])
    query = f"""
    SELECT ?city ?cityLabel ?countryLabel WHERE {{
      VALUES ?city {{ {city_values} }}
      ?city wdt:P17 ?country.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}"""

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    city_list = []
    for result in results['results']['bindings']:
        city_code = result['city']['value'].split('/')[-1]
        city = result['cityLabel']['value']
        country = result['countryLabel']['value']
        city_list.append({
            'city_code': city_code,
            'city': city,
            'country': country,
        })
    return city_list


def pair_profession_city(profession_path, city_path):
    profession_list = [json.loads(line) for line in open(profession_path)]
    city_list = [json.loads(line) for line in open(city_path)]

    data = []
    for profession in profession_list:
        for city in city_list:
            question = generate_question(profession['profession'], city['city'], city['country'])
            messages = [
                {'role': 'system',
                 'content': 'You are a helpful assistant designed to answer questions. Always begin responses with the direct answer.'},
                {'role': 'user',
                 'content': question},
            ]

            data.append({
                'profession_code': profession['profession_code'],
                'profession': profession['profession'],
                'city_code': city['city_code'],
                'city': city['city'],
                'country': city['country'],
                'question': question,
                'messages': messages,
            })
    # selected_idx = random.sample(range(len(wikidata)), 5000)
    # wikidata = [wikidata[i] for i in selected_idx]
    print(len(data))
    return data


def get_correct_answers(data):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)

    new_data = []
    for instance in tqdm(data):
        query = f"""
            SELECT DISTINCT ?person ?personLabel
            WHERE {{
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                ?person wdt:P106 wd:{instance['profession_code']};  
                wdt:P19 ?birthplace.
                ?birthplace (wdt:P131*) wd:{instance['city_code']}.

                OPTIONAL {{
                    ?person wikibase:sitelinks ?sitelinks.
                }}
            }}
            ORDER BY DESC (?sitelinks)
            LIMIT 100"""

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        answers = [result['personLabel']['value'] for result in results["results"]["bindings"]]
        answers = list(set(answers))
        if len(answers) < 20:
            continue

        instance['answers'] = answers
        new_data.append(instance)

    print(len(new_data))
    return new_data


def split_train_test(data):
    random.shuffle(data)

    train_size = 2000
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_path = 'wikidata_train_free.jsonl'
    with open(train_path, 'w') as f:
        for instance in train_data:
            f.write(json.dumps(instance) + '\n')
    print(len(train_data))
    print(f'Save train data to {train_path}')

    test_path = 'wikidata_test_free.jsonl'
    with open(test_path, 'w') as f:
        for instance in test_data:
            f.write(json.dumps(instance) + '\n')
    print(len(test_data))
    print(f'Save test data to {test_path}')


def get_where_query(input_filename):
    data = [json.loads(line) for line in open(input_filename)]
    new_data = []
    for index, instance in enumerate(data):
        for name in instance['answers'][:20]:
            where_question = f'Where was {name} born? Please answer with the name of the city.'

            messages = [
                {"role": "system",
                 "content": "You are a helpful and concise assistant. Answer questions accurately and start with the answer."},
                {"role": "user", "content": "Where was Donald Trump born? Please answer with the name of the city."},
                {"role": "assistant", "content": "New York City."},
                {"role": "user", "content": where_question}
            ]

            new_instance = copy.deepcopy(instance)
            new_instance['question'] = where_question
            new_instance['messages'] = messages
            new_instance.pop('answers')
            new_instance['name'] = name
            new_instance['result'] = True
            new_instance['index'] = index
            new_data.append(new_instance)

    print(len(new_data))
    output_filename = input_filename.replace('_free.jsonl', '_where.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


def get_profession_query(input_filename):
    data = [json.loads(line) for line in open(input_filename)]
    new_data = []
    for index, instance in enumerate(data):
        for name in instance['answers'][:20]:
            where_question = f'What is {name}\'s profession?'

            messages = [
                {"role": "system",
                 "content": "You are a helpful and concise assistant. Answer questions accurately and start with the answer."},
                {"role": "user", "content": "What is Donald Trump's profession?"},
                {"role": "assistant",
                 "content": "Donald Trump is an American politician, businessman, and media personality."},
                {"role": "user", "content": where_question}
            ]

            new_instance = copy.deepcopy(instance)
            new_instance['question'] = where_question
            new_instance['messages'] = messages
            new_instance.pop('answers')
            new_instance['name'] = name
            new_instance['result'] = True
            new_instance['index'] = index
            new_data.append(new_instance)

    print(len(new_data))
    output_filename = input_filename.replace('_free.jsonl', '_profession.jsonl')
    with open(output_filename, 'w') as f:
        for instance in new_data:
            f.write(json.dumps(instance) + '\n')
    print('Save data to', output_filename)


if __name__ == "__main__":
    # 生成名人较多的profession
    professions = get_professions_with_high_counts()
    profession_list = get_profession_labels(professions)
    output_path = 'profession.jsonl'
    with open(output_path, 'w') as f:
        for profession in profession_list:
            f.write(json.dumps(profession) + '\n')

    # 生成出生的名人较多的city
    cities = get_cities_with_high_counts()
    city_list = get_city_labels(cities)
    output_path = 'city.jsonl'
    with open(output_path, 'w') as f:
        for city in city_list:
            f.write(json.dumps(city) + '\n')

    # 生成raw wikidata
    data = pair_profession_city('profession.jsonl', 'city.jsonl')
    data_with_answers = get_correct_answers(data)
    split_train_test(data_with_answers)

    # 生成where query
    for filename in ['wikidata_train_free.jsonl', 'wikidata_test_free.jsonl']:
        get_where_query(filename)
        get_profession_query(filename)
