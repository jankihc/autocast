import openai
import numpy as np

# set OpenAI API key
openai.api_key = "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"

# define prompt and context
prompt = (f"Q: Which Republican presidential candidate will win the Ohio primary on 15 March?"
          f"A:")

context = (f"For more information on the candidates, the primary process, and the primary schedule see: Candidates, How the Presidential Primary Works, Schedule. Recommended Questions Who will win the 2016 US presidential election? Who will win the Republican Party nomination for the US presidential election? Will the Republican candidate for president win the party's nomination on the first ballot, at the party's convention in July?")

# define list of possible answers
possible_answers = ["Ted Cruz",
            " John Kasich",
            "Marco Rubio",
            "Donald Trump"]

# define function to calculate semantic similarity
def calculate_similarity(answer, context):
    # format context and answer
    input_text = f"Context: {context}\nAnswer: {answer}"
    # set parameters for semantic similarity
    params = {
        "model": "text-babbage-001",
        "documents": [input_text],
        "query": context
    }
    # call OpenAI's semantic similarity API
    similarity_score = calculate_semantic_similarity(answer, context)
    # extract the similarity score from the response
    #similarity_score = response["data"][0]["score"]
    return similarity_score

# define function to select the correct answer using self-consistent sampling
def select_answer(prompt, context, possible_answers, num_samples=10, temperature=0.5, similarity_threshold=0.7):
    # generate samples using OpenAI's API
    samples = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        #n=num_samples,
        temperature=temperature,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    # extract the generated answers from the samples
    print(samples)
    answers = [choice.text.strip() for choice in samples.choices]
    # calculate the semantic similarity score for each answer
    similarities = [calculate_similarity(answer, context) for answer in answers]
    print(similarities)
    print(answers)

    # selected_prompts = [prompt for prompt, score in zip(answers, similarities) if score > similarity_threshold]
    # if len(selected_prompts) > 0:
    #     # Select the most similar prompt
    #     prompt_text = selected_prompts[0]

    # calculate the probability of each answer based on its similarity score

    #####THIS IS WHERE I'M NOT SURE IF I'M DOING IT RIGHT TO CALCULATE PROBABILITY
    probabilities = [np.exp(similarity) / np.sum(np.exp(similarities)) for similarity in similarities]
    print(probabilities)

    # select the answer with the highest probability
    selected_answer_index_generated = np.argmax(probabilities)
    selected_answer_generated = answers[selected_answer_index_generated]

    similarities_orig = [calculate_similarity(answer, selected_answer_generated) for answer in possible_answers]
    selected_answer_index = np.argmax(similarities_orig)
    selected_answer = possible_answers[selected_answer_index]

    # return the selected answer and its probability
    return selected_answer, probabilities[selected_answer_index_generated]

# call the function to select the correct answer
selected_answer, probability = select_answer(prompt, context, possible_answers)
# print the selected answer and its probability
print(f"The correct answer is: {selected_answer}")
print(f"The probability of this answer is: {probability}")
