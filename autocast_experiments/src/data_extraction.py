import json
import pandas as pd

def main():
    autocast_questions = json.load(open('../competition/autocast_questions.json')) # from the Autocast dataset
    # test_questions = json.load(open('../competition/autocast_competition_test_set.json'))
    # test_ids = [q['id'] for q in test_questions]
    N = len(autocast_questions)
    q_arr = []
    a_arr = []
    qtype_arr = []
    choices_arr = []

    for i in range(N):
        q_arr.append(autocast_questions[i]["question"])
        choices_arr.append(autocast_questions[i]["choices"])
        a_arr.append(autocast_questions[i]["answer"])
        qtype_arr.append(autocast_questions[i]["qtype"])

    dictionary = {
        'q' : q_arr, 
        'choices': choices_arr,
        'a': a_arr,
        'qtype': qtype_arr
    }

    autocast_df = pd.DataFrame(dictionary)
    autocast_df.to_csv("autocast_questions_shortened.csv")
    print(autocast_df.head(5))
    return

if __name__ == "__main__":
    main()