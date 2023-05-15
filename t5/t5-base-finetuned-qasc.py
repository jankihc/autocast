import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, T5ForConditionalGeneration, Trainer, TrainingArguments

# read in the xlsx file
answers_csv = pd.read_csv('./competition/autocast_test_set_w_answers-2.csv', encoding='ISO-8859-1')
'''
answers = []
qtypes = []
for question in answers_csv.iterrows():
    question = question[1]
    if question['qtype'] == 't/f':
        ans_idx = 0 if question['answers'] == 'no' else 1
        ans = np.zeros(len(eval(question['choices'])))
        ans[ans_idx] = 1
        qtypes.append('t/f')
    elif question['qtype'] == 'mc':
        ans_idx = ord(question['answers']) - ord('A')
        ans = np.zeros(len(eval(question['choices'])))
        ans[ans_idx] = 1
        qtypes.append('mc')
    elif question['qtype'] == 'num':
        ans = float(question['answers'])
        qtypes.append('num')
    answers.append(ans)
'''


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

def countVec(ranswer, pred):
    if not ranswer or not pred:
        return 0
    # Create a CountVectorizer object to convert the sentences to vectors of word counts
    vectorizer = CountVectorizer(stop_words='english').fit_transform([ranswer, pred])
    # Calculate the cosine similarity between the two vectors
    cosine_sim = cosine_similarity(vectorizer[0], vectorizer[1])[0][0]
    # Print the cosine similarity
    return cosine_sim

import Levenshtein
def Levenshtein_vec(ranswer, pred):
    if not ranswer or not pred:
        return 0
    vectorizer = Levenshtein.distance(pred, ranswer)
    # Print the cosine similarity
    return vectorizer

def count_or_distilbert_vec(ranswer, pred):
    try:
        similarity = countVec(ranswer, pred)
        if similarity is not None:
            return similarity
    except:
        pass
    return distilbertVec(ranswer, pred)

import random

def extract_number(s):
    match = re.search(r'\d+(\.\d+)?', s)
    if match:
        num = float(match.group())
        if num > 1.0:
            return round(random.random(),7)
        else:
            return num
    else:
        return round(random.random(),7)


def brier_score(probabilities, answer_probabilities):
    return ((probabilities - answer_probabilities) ** 2).sum() / 2


from numpy import dot
from numpy.linalg import norm
import decimal
predictions = []
answers = []
qtypes = []

for ds_item in answers_csv.iterrows():
    ds_item = ds_item[1]
    prediction_arr = None
    answer_arr = None
    if ds_item['qtype'] == 'mc':
        answer =ds_item['answers']
        question = ds_item['question']
        choice =eval(ds_item['choices'])
        background = ds_item['background']
        background = background[:min(len(background), 64)]
        pred = get_response(question, background)
        opt = ord(answer) - ord('A')
        ranswer = choice[opt]
        prediction_acc = []
        for i in range(len(choice)):
            prediction_acc.append(Levenshtein_vec(choice[i], pred))# get a list of the accuray for every choice
        # Convert list to decimals
        prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
        summ = sum(prediction_acc)
        if summ == 0:
            summ = 1e-5
        # Divide each decimal by the sum of decimals
        prediction_acc = [float(x) / float(summ) for x in prediction_acc]
        prediction_arr = np.array(prediction_acc)
        answer_arr = np.zeros(len(choice))
        answer_arr[opt] = 1

    elif ds_item['qtype'] == 't/f':
        question = ds_item['question']
        answer =ds_item['answers']
        background = ds_item['background']
        # Print the context
        background = background[:min(len(background), 128)]
        pred = get_response(question+" Just answer 'yes' or 'no'.", background)
        #acc = countVec(answer, pred)
        prediction_acc = []
        prediction_acc.append(count_or_distilbert_vec("yes", pred))# get a list of the accuray for every choice
        prediction_acc.append(count_or_distilbert_vec("no", pred))# get a list of the accuray for every choice
        # Convert list to decimals
        prediction_acc = [decimal.Decimal(str(x)) for x in prediction_acc]
        summ = sum(prediction_acc)
        if summ == 0:
            summ = 1e-5
        # Divide each decimal by the sum of decimals
        prediction_acc = [float(x) / float(summ) for x in prediction_acc]
        prediction_arr = np.array(prediction_acc)
        #print(f"{acc:.2f}")
        if answer == 'yes':
            answer_arr = [1, 0]
            answer_arr = np.array(answer_arr)
        else:
            answer_arr = [0, 1]
            answer_arr = np.array(answer_arr)

    elif ds_item['qtype'] == 'num':
        question = ds_item['question']
        answer =ds_item['answers']
        background = ds_item['background']
        background = background[:min(len(background), 128)]
        pred = get_response(question, background)
        print("prediction")
        print(pred)
        pred_floats = extract_number(pred)
        print("extra")
        print(pred_floats)  
        #cos_sim = count_or_distilbert_vec(str(answer), str(pred_floats))
        #prediction_arr = float(cos_sim)
        #print("what is this")
        prediction_arr = pred_floats
        print(prediction_arr)
        answer_arr = answer
        print(answer_arr)
    predictions.append(prediction_arr)
    answers.append(answer_arr)
    qtypes.append(ds_item['qtype'])



tf_results, mc_results, num_results = [],[],[]
for p, a, qtype in zip(predictions, answers, qtypes):
    if qtype == 't/f':
        tf_results.append(brier_score(p, a))
    elif qtype == 'mc':
        mc_results.append(brier_score(p, a))
    else:
        num_results.append(np.abs(float(p) - float(a)))

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
import zipfile
import os

# Set the paths for the input and output files
input_file = os.path.join('submission', 'predictions.pkl')
output_file = 'submission.zip'

# Create a ZipFile object and open the output file in write mode
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:

    # Add the input file to the ZIP archive
    zipf.write(input_file)

# Print a message to indicate that the compression is complete
print(f'Successfully compressed {input_file} to {output_file}')
