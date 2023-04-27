import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, T5ForConditionalGeneration, Trainer, TrainingArguments
autocast_questions = json.load(open('./competition/autocast_questions.json', encoding='utf-8')) # from the Autocast dataset
test_questions = json.load(open('./competition/autocast_competition_test_set.json', encoding='utf-8'))
test_ids = [q['id'] for q in test_questions]
autocast_questions = [q for q in autocast_questions if q['status'] == 'Resolved']

filtered_train_data = [example for example in autocast_questions if example['id'] not in test_ids and example['answer'] is not None]

import random

selected_data = []

# dictionary to keep track of number of selected questions for each type
selected_counts = {'mc': 0, 'num': 0, 't/f': 0}
desired_counts = {'mc': 200, 'num': 100, 't/f': 500}

# loop over examples in filtered_train_data
for example in filtered_train_data:
    # check if this example's qtype has already been selected enough times
    if selected_counts[example['qtype']] >= desired_counts[example['qtype']]:
        continue
        
    # select this example
    selected_data.append(example)
    selected_counts[example['qtype']] += 1
    
    # check if we've selected enough examples of each type
    if all(count >= desired_counts[qtype] for qtype, count in selected_counts.items()):
        break


import re
CLEANR = re.compile('<.*?>') 

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-qasc")
model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-qasc")

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def get_response(question, context, max_length=64):
  input_text = 'question: %s  context: %s' % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return cleanhtml(tokenizer.decode(output[0]))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModel
'''
tokenizer_acc = AutoTokenizer.from_pretrained("distilbert-base-cased")
model_acc = AutoModel.from_pretrained("distilbert-base-cased")

def distilbertVec(ranswer, pred):
    # Tokenize and encode the sentences
    real_answer_tokens = tokenizer_acc.encode(ranswer, return_tensors="pt")
    prediction_tokens = tokenizer_acc.encode(pred, return_tensors="pt")

    # Generate fixed-length vector representations of the sentences
    with torch.no_grad():
        real_answer_vec = model_acc(real_answer_tokens)[0][:,0,:]
        prediction_vec = model_acc(prediction_tokens)[0][:,0,:]

    # Compute the cosine similarity between the two sentence vectors
    similarity = cosine_similarity(real_answer_vec, prediction_vec)
    return similarity[0][0]   
'''
def countVec(ranswer, pred):
    # Create a CountVectorizer object to convert the sentences to vectors of word counts
    vectorizer = CountVectorizer().fit_transform([ranswer, pred])
    # Calculate the cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    # Print the cosine similarity
    return cosine_sim

import re

def extract_float(s):
    try:
        return float(re.search(r'\d+\.\d+', s).group())
    except AttributeError:
        return float(0)

def brier_score(probabilities, answer_probabilities):
    return ((probabilities - answer_probabilities) ** 2).sum() / 2

from numpy import dot
from numpy.linalg import norm
import decimal
predictions = []
answers = []
qtypes = []
for question_idx, ds_item in enumerate(selected_data):
    prediction_arr = None
    answer_arr = None
    if ds_item['qtype'] == 'mc':
        answer =ds_item['answer']
        question = ds_item['question']
        choice = ds_item['choices']  
        background = ds_item['background']
        background = background[:min(len(background), 64)]
        # Print the context
        #print(answer)
        print(question)
        print(background)
        print(answer)
        pred = get_response(question, background)
        opt = ord(answer) - ord('A')
        ranswer = choice[opt]
        print(ranswer)
        prediction_acc = []
        print("pred"+pred)
        for i in range(len(choice)):
            prediction_acc.append(countVec(choice[i], pred))# get a list of the accuray for every choice
            print("choice "+choice[i])
        # Convert list to decimals
        print(prediction_acc)
        prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
        summ = sum(prediction_acc)
        if summ == 0:
            summ = 1e-5
        # Divide each decimal by the sum of decimals
        prediction_acc = [float(x) / float(summ) for x in prediction_acc]
        prediction_arr = np.array(prediction_acc)
        answer_arr = np.zeros(len(choice))
        answer_arr[opt] = 1
        print(prediction_arr)
        print(answer_arr)
        
    elif ds_item['qtype'] == 't/f':#G 28, G30
        question = ds_item['question']
        answer =ds_item['answer']
        background = ds_item['background']
        # Print the context
        print(question)
        print(background)
        print(answer)
        background = background[:min(len(background), 128)]
        pred = get_response(question+" Just answer 'Yes' or 'No'.", background)
        print(pred)
        #acc = countVec(answer, pred)
        prediction_acc = []
        prediction_acc.append(countVec("yes", pred))# get a list of the accuray for every choice
        prediction_acc.append(countVec("no", pred))# get a list of the accuray for every choice
        # Convert list to decimals
        print(prediction_acc)
        prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
        summ = sum(prediction_acc)
        if summ == 0:
            summ = 1e-5
        # Divide each decimal by the sum of decimals
        prediction_acc = [float(x) / float(summ) for x in prediction_acc]
        prediction_arr = np.array(prediction_acc)
        #print(f"{acc:.2f}")
        print(prediction_arr)
        if answer == 'yes':
            answer_arr = [1, 0]
            answer_arr = np.array(answer_arr)
        else:
            answer_arr = [0, 1]
            answer_arr = np.array(answer_arr)
        print(answer_arr)
        #print(f"{distilbertVec(answer, pred):.2f}")
    
    elif ds_item['qtype'] == 'num':
        question = ds_item['question']
        answer =ds_item['answer']
        background = ds_item['background']
        # Print the context
        print(question)
        print(answer)
        print(background)
        background = background[:min(len(background), 128)]
        pred = get_response(question, background)
        print(pred)
        pred_floats = extract_float(pred)
        print(pred_floats)  
        cos_sim = countVec(str(answer), str(pred_floats))
        prediction_arr = float(cos_sim)
        print(prediction_arr)
        answer_arr = answer
    predictions.append(prediction_arr)
    answers.append(answer_arr)
    qtypes.append(ds_item['qtype'])
    print(predictions)
    print(answers)
    print(qtypes)

tf_results, mc_results, num_results = [],[],[]
for p, a, qtype in zip(predictions, answers, qtypes):
    if qtype == 't/f':
        tf_results.append(brier_score(p, a))
    elif qtype == 'mc':
        mc_results.append(brier_score(p, a))
    else:
        num_results.append(np.abs(p - a))

if not os.path.exists('submission'):
    os.makedirs('submission')
with open(os.path.join('submission', 'train_predictions.pkl'), 'wb') as f:
    pickle.dump(predictions, f, protocol=2)
print("Prediction saved!")
print()

# Get the perfroamcen and combined metric
performance = f"T/F: {np.mean(tf_results)*100:.2f}, MCQ: {np.mean(mc_results)*100:.2f}, NUM: {np.mean(num_results)*100:.2f}"
combined_metric = f"Combined Metric: {(np.mean(tf_results) + np.mean(mc_results) + np.mean(num_results))*100:.2f}"
print(performance)
print(combined_metric)
with open(os.path.join('submission', 'report.txt'), 'w') as f:
    f.write(str(performance) + "\n" + str(combined_metric))
    
'''
import decimal
def cal_posb():
    predictions = []
    answers = [] 
    for question_idx, ds_item in enumerate(filtered_train_data):
        if ds_item['qtype'] == 'mc':
            answer =ds_item['answer']
            question = ds_item['question']
            choice = ds_item['choices']  
            background = ds_item['background']
            background = background[:min(len(background), 128)]
            # Print the context
            #print(answer)
            pred = get_response(question, background)
            choices = choice.values.tolist()
            opt = ord(answer) - ord('A')
            prediction_acc = []
            for i in range(len(choices)):
                prediction_acc[i] = countVec(choices[i], pred)# get a list of the accuray for every choice
            prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
            summ = sum(prediction_acc)
            # Divide each decimal by the sum of decimals
            prediction_acc = [x / summ for x in prediction_acc]
            prediction_arr = np.array(prediction_acc)
            answer_arr = np.zeros(len(choices))
            answer_arr[opt] = 1
        predictions.append(prediction_arr)
        answers.append(answer_arr)

import numpy as np
predicted_probs = np.array(a)
true_probs = np.array([0, 0, 1, 0])
score = brier_score(predicted_probs, true_probs)#the list a, [1, 0, 0, 0] for example
print(score)


import string 
def calculate_answer_probability(answer_arr, choice_arr, qtype):
    """
    Give the answer array, calculate the possibilities of each answer given the choices and answer_arr
    """
    # Process the answer
    if qtype == "t/f":
        # unique: 2x1 array
        # counts: 2x1 array
        choice_arr_tf = ["yes", "no"]
        N = len(answer_arr)
        unique, counts = np.unique(answer_arr, return_counts=True)
        choice_len = len(choice_arr_tf)
        new_counts = np.zeros(choice_len)
        for i in range(choice_len):
            curr_choice = choice_arr_tf[i]
            if curr_choice not in unique:
                new_counts[i] = 0
            else:
                new_counts[i] = counts[curr_choice == unique]

        # Probability of yes/no
        return new_counts / N
    
    elif qtype == "mc":
        # Total number of prediciton
        # Should be 10
        # choice_arr = process_choice(choice_arr)
        N = len(answer_arr)
        unique, counts = np.unique(answer_arr, return_counts=True)
        choice_len = len(choice_arr)
        # Generate answer choice
        answer_choice = list(string.ascii_uppercase)[:choice_len]
        # Create a new counts array to hold the prob of all answers
        new_counts = np.zeros(len(choice_arr))
        for i in range(choice_len): 
            curr_choice = answer_choice[i]
            if curr_choice not in unique: 
                new_counts[i] = 0
            else:
                new_counts[i] = counts[curr_choice == unique]
 
        # Probability of each answer in an array
        return new_counts / N

    elif qtype == "num":
        # Number questions, return the number only  
        return answer_arr[0]
        '''