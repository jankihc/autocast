#cosine similarity
import math
from sklearn.metrics.pairwise import cosine_similarity

import openai 
import pandas as pd
import time
import argparse
import os
import json
import collections

openai.api_key = os.getenv('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')

parser = argparse.ArgumentParser()

# if an argument is passed in as True, we do it
parser.add_argument("--Codex_Few_Shot")

parser.add_argument("--GPT3_CoT_One_Shot")

parser.add_argument("--Do_MATH")

parser.add_argument("--Do_Courses")

#args = parser.parse_args()
args, unknown = parser.parse_known_args()

few_shot_examples_desired = 5
codex_engine = "code-davinci-002"
gpt3_engine = "text-davinci-002"
engine_temperature = 0
engine_topP = 0
few_shot_max_tokens = 256
gpt3_CoT_max_tokens = 1000
codex_time_delay = 3
gpt3_time_delay = 1
CoT = "Let's think step by step."

f = open('../competition/autocast_questions.json')
json_obj = json.load(f)

def get_details_from_autocast_json_using_index(idx):
    # for i in json_obj:
    #     question = i['question']
    return json_obj[idx]['question'], json_obj[idx]['choices'], json_obj[idx]['answer'], json_obj[idx]['background']

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def get_cosine_similarity(q1, q2):
    return round(cosine_similarity([q1], [q2])[0][0], 8)

# define function to select the correct answer using self-consistent sampling
def select_answer(prompt, num_samples=10, temperature=0.5, similarity_threshold=0.7):
    # generate samples using OpenAI's API
    samples = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=num_samples,
        temperature=temperature,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    # extract the generated answers from the samples
    answers = [choice.text.strip() for choice in samples.choices]
    summ = 0
    for choice in collections.Counter(answers).values():
      summ += choice
    # count the frequency of each answer
    answer_counts = collections.Counter(answers)
    # get the most common answer and its count
    generated_answer = answer_counts.most_common(1)[0][0]
    frequency_of_generated_answer = answer_counts.most_common(1)[0][1]

    # return the selected answer and its probability
    return generated_answer, frequency_of_generated_answer/summ

# def get_few_shot_output(few_shot_input):
#     start = time.time()
#     time.sleep(codex_time_delay) #to avoid an openai.error.RateLimitError
#     few_shot_output = openai.Completion.create(engine = codex_engine, 
#                                             prompt = few_shot_input, 
#                                             max_tokens = few_shot_max_tokens, 
#                                             temperature = engine_temperature, 
#                                             top_p = engine_topP, output_scores=True)['choices'][0]['text']
#     print('Codex API call time: ' + str(time.time()-start) + '\n')
#     #print(few_shot_output)
#     return few_shot_output

def get_few_shot_input(orig_question, choices, similarity_list):
    # get similar_question, similar_answer, similar_chatgpt_answer, originial_question
    few_shot_input = ''
    # try with 3 and 5
    for i in range(3):
        #this is for multiple choice questions
        question, choices, answer, background = get_details_from_autocast_json_using_index(similarity_list[i])
        few_shot_input += '\nQuestion: ' + question
        few_shot_input += '\nChoices: ' + choices
        few_shot_input += '\nAnswer: ' + answer
        few_shot_input += '\nBackground' + background

    few_shot_input += '\nQuestion: ' + orig_question
    few_shot_input += '\nOnly answer with alphabet, do not explain: ' + choices
    return few_shot_input

def run(question, choices):
    #compute embedding
    test_question_embedding = get_embedding(question)

    #do we compute embeddings beforehand for training set?
    #get similar questions
    #assuming we have embeddings in json
    dict_of_similarities = {}
    for idx in json_obj:
        #check for resolved
        if idx['status'] == 'resolved':
            train_embedding = get_embedding(idx['question'])
            dict_of_similarities[idx] = get_cosine_similarity(test_question_embedding, train_embedding)
            dict_of_similarities = dict(sorted(dict_of_similarities.items(), key=lambda item: item[1], reverse = True))
        
    similar_question_indices = list(dict_of_similarities.keys())

    #get few shot input
    few_shot_input = get_few_shot_input(question, choices, similar_question_indices)
    
    #give it as input to few shot and check the answers
    answer, probability = select_answer(few_shot_input)
    
    return answer, probability
