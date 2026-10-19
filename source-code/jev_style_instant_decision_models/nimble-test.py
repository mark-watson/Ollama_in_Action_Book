import ollama

response = ollama.systemone(
    model='nimble:9b',
    state='The quick brown fox jumps over the lazy dog. This is a longer state that contains multiple sentences. The fox is very agile and can jump over obstacles with ease. The lazy dog prefers to sleep all day long. The sun is shining brightly in the sky, and birds are chirping melodiously. The world is full of wonderful things to discover and explore.',
    questions={
        'says_hello': {
            'type': 'noul',
            'instructions': 'Does the state text contain a greeting?',
            'criteria': {
                'true': 'The state text contains a greeting.',
                'false': 'The state text does not contain a greeting.',
            },
        },
        'mentions_fox': {
            'type': 'noul',
            'instructions': 'Does the state mention a fox?',
            'criteria': {
                'true': 'The state mentions a fox.',
                'false': 'The state does not mention a fox.',
            },
        },
        'mentions_dog': {
            'type': 'noul',
            'instructions': 'Does the state mention a dog?',
            'criteria': {
                'true': 'The state mentions a dog.',
                'false': 'The state does not mention a dog.',
            },
        },
        'is_positive': {
            'type': 'noul',
            'instructions': 'Is the overall sentiment of the state positive?',
            'criteria': {
                'true': 'The sentiment is positive.',
                'false': 'The sentiment is not positive.',
            },
        },
        'has_obstacles': {
            'type': 'noul',
            'instructions': 'Does the state mention any obstacles?',
            'criteria': {
                'true': 'The state mentions obstacles.',
                'false': 'The state does not mention obstacles.',
            },
        },
        'has_birds': {
            'type': 'noul',
            'instructions': 'Does the state mention birds?',
            'criteria': {
                'true': 'The state mentions birds.',
                'false': 'The state does not mention birds.',
            },
        },
        'has_sun': {
            'type': 'noul',
            'instructions': 'Does the state mention the sun?',
            'criteria': {
                'true': 'The state mentions the sun.',
                'false': 'The state does not mention the sun.',
            },
        },
        'contains_animals': {
            'type': 'noul',
            'instructions': 'Does the state contain references to animals?',
            'criteria': {
                'true': 'The state contains animal references.',
                'false': 'The state does not contain animal references.',
            },
        },
        'length_check': {
            'type': 'noul',
            'instructions': 'Is the state text more than 50 characters long?',
            'criteria': {
                'true': 'The state is more than 50 characters.',
                'false': 'The state is not more than 50 characters.',
            },
        },
        'has_world': {
            'type': 'noul',
            'instructions': 'Does the state mention the world?',
            'criteria': {
                'true': 'The state mentions the world.',
                'false': 'The state does not mention the world.',
            },
        },
    },
)
print("=" * 60)
print("NIMBLE TEST RESULTS")
print("=" * 60)
print(f"Question: says_hello")
print(f"Answer: {response.answers['says_hello'].noul}")
print("-" * 60)
print("All Answers:")
print("-" * 60)
for question_name, answer_obj in response.answers.items():
    print(f"  {question_name}:")
    print(f"    {answer_obj.noul}")
print("=" * 60)