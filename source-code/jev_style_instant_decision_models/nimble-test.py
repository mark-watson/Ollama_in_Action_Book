import ollama

response = ollama.systemone(
    model='nimble:9b',
    state='Hello World',
    questions={
        'says_hello': {
            'type': 'noul',
            'instructions': 'Does the state text contain a greeting?',
            'criteria': {
                'true': 'The state text contains a greeting.',
                'false': 'The state text does not contain a greeting.',
            },
        },
    },
)
print(response.answers['says_hello'].noul)
print(response.answers)
