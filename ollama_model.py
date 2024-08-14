import ollama


def ollama_response(question):
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': question,
        },
    ])
    return response['message']['content']


if __name__ == '__main__':
    print(ollama_response("Answer question What is the capital of Poland? base your answer on text bellow. Text: \n "
                          "Poland is a country in Europe. Its capital is Warsaw."))