import pandas as pd
import numpy as np
import openai
import pickle
import json
from tqdm import tqdm
from math import isnan
import time
from openai.embeddings_utils import get_embedding, cosine_similarity
import string
import os
import warnings
warnings.filterwarnings('ignore')

# GLOBAL VARIABLES -
# Can change to improve performance
#####
# Change this openai.api_key to your key
openai.api_key = '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'
#####
autocast_train_with_dates = pd.read_csv('autocast_questions_shortened_with_dates.csv')

EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_MODEL_answer = 'gpt-3.5-turbo'
embedding_cache_path = 'data/embedding_cache.pkl'
gpt3_engine = "text-davinci-002"
few_shot_max_tokens = 5
engine_topP = 1
engine_temperature = 0
codex_time_delay = 3
# top k exampels for few-shot learning
k = 6
rate_limit_per_minute = 2
# delay request in seconds to avoid Openai.RateLimitError
delay = 60 / rate_limit_per_minute
# Obtain number of questions on the dataset
desire_index = 2


# Brier score for the performance on the train set
def brier_score(probabilities, answer_probabilities):
    return ((probabilities - answer_probabilities) ** 2).sum() / 2

def process_choice(choice_arr):
    choice_arr = choice_arr.replace("[", "").replace("]", "")
    choice_arr = np.array(choice_arr.split("'"))
    new_choice_arr = []
    for i in range(len(choice_arr)):
    # print(i, arr[i])
        if choice_arr[i] != ', ' and choice_arr[i] != '':
            new_choice_arr.append(choice_arr[i])

    return new_choice_arr


def process_answer(answer_arr, qtype, choice_arr):
    """
    Process the answers given from GPT-3 model to simple format
    """
    N = len(answer_arr)
    new_answer_arr = []
    if qtype == "t/f":
        for i in range(N):
            curr_answer = answer_arr[i].lower()
            if "yes" in curr_answer:
                new_answer_arr.append("yes")
            else:
                new_answer_arr.append("no")
        return new_answer_arr
    
    elif qtype == "mc":
        # new_choice_arr = process_choice(choice_arr)

        choice_len = len(choice_arr)
        # print("Length of choice arr", choice_len)
        answer_choice = list(string.ascii_uppercase)[:choice_len]
        # print("answer choice", answer_choice)
        for i in range(N):
            curr_answer = answer_arr[i] 
            for j in range(len(answer_choice)):
                curr_choice = choice_arr[j]

                if curr_choice in curr_answer:
                    new_answer_arr.append(answer_choice[j])
        return new_answer_arr
    
        # print("Current choice:", choice_arr)
        # print("Type of choice", type(choice_arr))
        # return choice_arr
    
    elif qtype == "num":   
        return answer_arr
    

# Answer format 
# [Prob of answer A, Prob of answer B, etc...]
# sum up to 1
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
        N = len(answer_arr)
        answer = []
        for str in answer_arr:
            number_str = ""
            for char in str: 
                if char.isdigit():
                    number_str += char
            try:
                number = float(number_str)
            except ValueError:
                number = 0.

            answer.append(number)
        
        unique, counts = np.unique(answer, return_counts=True)
        
        res = max(counts / N)
        return res

def get_few_shot_output(input):
    few_shot_output = openai.Completion.create(engine = gpt3_engine, 
                                            prompt = input, 
                                            max_tokens = few_shot_max_tokens, 
                                            temperature = engine_temperature, 
                                            top_p = engine_topP)['choices'][0]['text']
    return few_shot_output

# Get GPT answers from the prompt
def evaluate(eval_prompt, examples, role):
    curr_prompt = f"""    
        {examples}
        We then have the this question: 
        {eval_prompt}
        """
    # print(curr_prompt)
    role_content = "You are " + role
    answers = []
    # Perform self-consistency sampling 
    # By asking the GPT-3 model 10 times
    # Delay request API call to avoid openai.RateLimitError
    time.sleep(delay)
    # Set limit to 5 tokens to get a short answer
    # temperature = 0 to control the output
    response = openai.ChatCompletion.create(
            model=EMBEDDING_MODEL_answer,
            temperature=0.5,
            max_tokens=20,
            n=10,
            messages = [
                {"role": "system", "content": role_content},
                {"role": "user", "content" : curr_prompt},
            ]
        )
    # Assistant answer
    for elem in response['choices']:
        answers.append(elem['message']["content"].strip())

    return answers

# Function to create prompt for GPT3
# Based on different type of questions, different prompt is invoked
def create_prompt(qtype, choice_original, similarity_list, question, autocast_df, curr_choices_list):
    role = question + " - Who would be the best person to answer the above question? Don't explain, answer in 2 to 6 words."
    prompt = ""
    qtype_prompt = ""
    if qtype == 't/f':
        qtype_prompt += "answer yes or no only"
        eval_prompt = f"""
            Q: {question} \n
            A: Don't explain, make a prediction, {qtype_prompt}
            """
    elif qtype == "mc":
        qtype_prompt += "State only answers in each bracket without any justification."
        eval_prompt = f"""
                Q: {question} \n
                Choices: {curr_choices_list} \n
                A: Don't explain, make a prediction from one of the choices above. {qtype_prompt}
                """
    elif qtype == "num":
        qtype_prompt += "print out that prediction number only without any justification, do not explain."
        eval_prompt = f"""
                Q: {question} \n
                {choice_original} \n
                Based on the max, min, and standard deviation, make a prediction. {qtype_prompt}
                """
    prompt += "Let's take these question, choices, and answer as a pattern and trends for your prediction: \n"
    for i in similarity_list:
        prompt += f"""
                Q: {autocast_df['q'].loc[i]} \n
                Choices: {autocast_df.loc[i]['choices']} \n
                A: {autocast_df.loc[i]['a']} \n
                """

    return eval_prompt, prompt, role

# Return top k similarity questions to do few-shot learning
def top_k_similarity(question, question_index, corpus, k):
    """
    question: question embedding
    corpus: corpus embedding
    Return top k similarity of the questions based on embedding
    """
    cosine_list = []
    for i in range(len(corpus)):
        #skip the questions with publish_time > closed_time 
        if (autocast_train_with_dates[question_index][5] > autocast_train_with_dates[i][4]):
            continue
        curr_corpus_question = corpus[i][0]
        curr_corpus_index = corpus[i][1]
        # print(len(question))
        # print(len(curr_corpus_question))
        cosine = cosine_similarity(question, curr_corpus_question)
        cosine_list.append((cosine, curr_corpus_index))
    
    # Return top k similar question
    return sorted(cosine_list, key=lambda tup: tup[0], reverse=True)[:k]


################################################################################################
# Section to obtain baseline results given in the example_submisison.ipynb from competition github
def calibrated_random_baseline_model(question):
    if question['qtype'] == 't/f':
        pred_idx = np.argmax(np.random.random(size=2))
        pred = np.ones(2)
        pred[pred_idx] += 1e-5
        return pred / pred.sum()
    elif question['qtype'] == 'mc':
        pred_idx = np.argmax(np.random.random(size=len(question['choices'])))
        pred = np.ones(len(question['choices']))
        pred[pred_idx] += 1e-5
        return pred / pred.sum()
    elif question['qtype'] == 'num':
        return 0.5

def get_baseline_results():
    autocast_questions = json.load(open('competition/autocast_questions.json')) # from the Autocast dataset
    test_questions = json.load(open('competition/autocast_competition_test_set.json'))
    test_ids = [q['id'] for q in test_questions]

    preds = []
    answers = []
    qtypes = []
    for question in autocast_questions:
        if question['id'] in test_ids: # skipping questions in the competition test set
            continue
        if question['answer'] is None: # skipping questions without answer
            continue
        preds.append(calibrated_random_baseline_model(question))
        if question['qtype'] == 't/f':
            ans_idx = 0 if question['answer'] == 'no' else 1
            ans = np.zeros(len(question['choices']))
            ans[ans_idx] = 1
            qtypes.append('t/f')
        elif question['qtype'] == 'mc':
            ans_idx = ord(question['answer']) - ord('A')
            ans = np.zeros(len(question['choices']))
            ans[ans_idx] = 1
            qtypes.append('mc')
        elif question['qtype'] == 'num':
            ans = float(question['answer'])
            qtypes.append('num')
        answers.append(ans)
    print(preds, answers)
    return preds, answers
################################################################################################

# Few shot learning
def main():

    print("---------- Obtain Performance on Random Baseline model ----------")
    baseline_prediction, baseline_answers = get_baseline_results()
    print("Done!")
    print()


    print("---------- Get Question CSV ----------")
    df = pd.read_csv('autocast_experiments/src/autocast_questions_shortened.csv')
    n, _ = df.shape
    print("Total number of questions:", n)
    print()

    print("---------- Get Question Embedding ----------")
    embedding_cache_path = "autocast_experiments/src/data/recommendations_embeddings_cache.pkl"

    # load the cache if it exists, and save a copy to disk
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)

    # define a function to retrieve embeddings from the cache if present, and otherwise request via the API
    def embedding_from_string(
        i: int,
        string: str,
        model: str = EMBEDDING_MODEL,
        embedding_cache=embedding_cache
    ) -> list:
        """Return embedding of given string, using a cache to avoid recomputing."""
        if (string, model) not in embedding_cache.keys():
            # print(f"Question {i} not in cache, adding embedding")
            embedding_cache[(string, model)] = get_embedding(string, model)
            with open(embedding_cache_path, "wb") as embedding_cache_file:
                pickle.dump(embedding_cache, embedding_cache_file)
        # else:
            # print(f"Question {i} already in cache")
        return embedding_cache[(string, model)]
    
    # Question embedding
    print("Currently saving embedding...")
    count = 0
    for i in tqdm(range(n)):
        curr_q = df['q'].iloc[i]
        embedding = embedding_from_string(i, curr_q)
        count += 1
    print(f"Done saving embedding for {count} autocast question!")
    print()

    print("---------- Calculate Cosine Similarity ----------")
    ####
    # Change this directory to the current directory of your autocast_questions.json
    autocast_questions = json.load(open('competition/autocast_questions.json')) # from the Autocast dataset
    # Change this directory to the current directory of your autocast_test_set.json
    test_questions = json.load(open('competition/autocast_competition_test_set.json'))
    test_ids = [q['id'] for q in test_questions]
    ####
    print("Get the embedding for questions not in test set...")
    # Get the corpus embedding
    corpus_embedding = []
    for i in range(n):
        curr_q_id = autocast_questions[i]['id']
        # Ignore test question
        if curr_q_id in test_ids:
            continue
        else:
            curr_q = df['q'].iloc[i]
            # Time to retrieve embedding should be low because retrieving from cache
            curr_q_embedding = embedding_from_string(i, curr_q)
            corpus_embedding.append((curr_q_embedding, i))

    print("Length of corpus embedding:", len(corpus_embedding))
    print("Done!")  
    print()


    print("---------- Retrieving index of samples ----------")
    tf_index = []
    mc_index = []
    num_index = []

    for i in range(n):
        curr_qtype = df['qtype'].iloc[i]

        if curr_qtype == 't/f':
            if len(tf_index) == desire_index:
                continue
            else:
                tf_index.append(i)

        elif curr_qtype == 'mc':
            if len(mc_index) == desire_index:
                continue
            else:
                mc_index.append(i)
        
        elif curr_qtype == 'num':
            if len(num_index) == desire_index:
                continue
            else: 
                num_index.append(i)

        if len(num_index) == desire_index and len(tf_index) == desire_index and len(mc_index) == desire_index:
            break

    # print(f"True/False first {desire_index} questions: {tf_index}")
    # print()
    # print(f"Multiple Choice first {desire_index} questions: {mc_index}")
    # print()
    # print(f"Numerical first {desire_index} questions: {num_index}")
    # print()

    # print(df['qtype'].iloc[num_index[0]])
    print("Done!")
    print()

    # quesitons_indexes = [tf_index, mc_index, num_index]
    # quesitons_indexes = [tf_index]
    quesitons_indexes = [num_index]
    # quesitons_indexes = [mc_index]

    # types = ["tf", "mc", "num"]
    # types = ["tf"]
    # types = ["mc"]
    types = ["num"]


    print("---------- Few Shot Learning ----------")
    # Few shot learning examples
    # Only perform on autocast test question

    for j in range(len(quesitons_indexes)):
        predictions = []
        answers = []
        qtypes = []
        
        baseline_preds = []
        baseline_ans = []

        index_arr = quesitons_indexes[j]
        print()
        print(f"Calculating {types[j]} questions")
        for i in tqdm(index_arr):
            baseline_preds.append(baseline_prediction[i])
            baseline_ans.append(baseline_answers[i])

            curr_q_id = autocast_questions[i]['id'] 
            curr_q_ans = autocast_questions[i]['answer']
            
            # Skip question in the test set
            if curr_q_id in test_ids:
                continue
            # Skip question without answer
            if curr_q_ans is None:
                continue

            curr_q = df['q'].iloc[i]
            # print("Train question:", curr_q)
            curr_qtype = df['qtype'].iloc[i]
            # print("Train question type:", curr_qtype)
            # print("Train answer:", curr_q_ans)

            curr_choices = df['choices'].iloc[i]
            # print("Choices original:", curr_choices)
            # print()
            curr_choices_list = process_choice(curr_choices)
            # print("Choices after processed:", curr_choices_list)

            # Skip questions with more than 15 options:
            # Could be error due to data process / extraction
            if len(curr_choices_list) > 15: 
                continue

            # print("Length of choices:", len(curr_choices_list))

            # Get the performance on the train set
            if curr_qtype == 't/f':
                # ans_idx = 0 if curr_q_ans == 'no' else 1
                # ans = np.zeros(len(curr_choices_list))
                # ans[ans_idx] = 1
                if curr_q_ans == 'no':
                    ans = [0., 1.0]
                else:
                    ans = [1.0, 0.]
                qtypes.append('t/f')
            elif curr_qtype == 'mc':
                ans_idx = ord(curr_q_ans) - ord('A')
                ans = np.zeros(len(curr_choices_list))
                ans[ans_idx] = 1
                qtypes.append('mc')
            elif curr_qtype == 'num':
                ans = float(curr_q_ans)
                qtypes.append('num')
            answers.append(ans)

            # print("Type of choices:", type(curr_choices))
            # Get the current embeddings
            curr_q_embedding = embedding_from_string(i, curr_q)
            # print("Current test embedding:", len(curr_q_embedding))
            # Get the top k index questions similar to the test questions
            top_k_q_index = top_k_similarity(curr_q_embedding, i, corpus_embedding, k)
            # print("Index of similarity:", top_k_q_index)
            # Create the prompt
            # Get the index list
            sim_list = []
            for elem in top_k_q_index:
                _, index = elem
                sim_list.append(index)

            # Create the prompt
            if curr_qtype == "num":
                test_q_prompt = create_prompt(curr_qtype, curr_choices, sim_list, curr_q, df, curr_choices_list)
            else:
                test_q_prompt = create_prompt(curr_qtype, i, sim_list, curr_q, df, curr_choices_list)
            # Get the answer
            test_q_output = evaluate(test_q_prompt[0], test_q_prompt[1], test_q_prompt[2])
            # print("Current GPT-3 Result:", test_q_output)
            new_output = process_answer(test_q_output, curr_qtype, curr_choices_list)
            # print("New output processed:", new_output)
            pred = calculate_answer_probability(new_output, curr_choices_list, curr_qtype) 
            # print("Prediction probability:", pred)
            # print("Answer probability", ans)
            predictions.append(pred)
            # print()
            # print()

        print(f"    Saving Predictions for {types[j]} questions")
        if not os.path.exists(f'submission_{desire_index}_{types[j]}'):
            os.makedirs(f'submission_{desire_index}_{types[j]}')

        # with open(os.path.join('submission', f'{types[j]}_train_predictions.pkl'), 'wb') as f:
        #     pickle.dump(predictions, f, protocol=2)

        print(f"    Calculate performance of model with {types[j]} questions on train set")
        results = []
        baseline_results = []
        # print("Current types:", types[j] == 'num')
        if types[j] == "num": 
            for p, a in zip(predictions, answers):
                results.append(np.abs(p - a))

            for p, a in zip(baseline_preds, baseline_ans):
                print("Printing p and a", p, a)
                print(np.abs(p - a))
                baseline_results.append(np.abs(p - a))

        else:
        # if types[j] == "mc" or types[j] == 'tf':
            for p, a in zip(predictions, answers):
                brier = brier_score(p, a)
                if isnan(brier):
                    results.append(0)
                else:
                    results.append(brier)
            
            for p, a in zip(baseline_preds, baseline_ans):
                baseline_results.append(brier_score(p, a))


        print(f"    Writing Performance for {types[j]} questions")

        if len(results) == 0:
            mean_res = 0.
        else:
            mean_res = np.mean(results)*100

        print(baseline_results)

        if len(baseline_results) == 0:
            mean_base = 0.
        else:
            print("in here!")
            mean_base = np.mean(baseline_results)*100

        # print(mean_base)
        # bug with the numerical questions, no idea
        # np.mean() returns an array instead of a number for the numerical questions
        # [mean_value, mean_value]
        if types[j] == 'num':
            mean_base = mean_base[0]

        with open(os.path.join(f'submission_{desire_index}_{types[j]}', f'{types[j]}_report.txt'), 'w+') as f:
            f.write(f"GPT Model performance on {types[j]} questions: {mean_res:.2f} \n")
            f.write(f"Baseline model performance on {types[j]} questions: {mean_base:.2f}")
        print(f"    Done!")
    

if __name__ == "__main__":
    main()