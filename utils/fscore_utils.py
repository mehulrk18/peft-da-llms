import glob
import json
import logging
import os
import sys
from typing import List

import pandas as pd
# import openai
# from openai import OpenAIError  #, OpenAI # BadRequestError
from openai import BadRequestError, OpenAI  # above two lines replacing this line


def get_openai_key():
    openai_tok = os.environ.get("OPENAI_API_KEY")
    assert openai_tok and openai_tok != "<openai_token>", "OpenAI token is not defined"
    return openai_tok.strip()


def get_wiki_topic(query: list) -> List:
    response = None
    wiki_topics = []
    num_rate_errors = 0

    openai_client = OpenAI(api_key=get_openai_key()) # below line replacing this line
    # openai_client = openai
    prompts = [
        (
            f"Your task it to Map the provided text to an existing wikipedia article that describes it the best. Please only output the name of the article name make sure that the word is a valid wikipedia entry."
            f"\n Text: {q} \n Wikipedia article name: "
        )
        for q in query
    ]
    for prompt in prompts:
        received = False
        while not received:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=10,
                    temperature=0.2,
                )
                received = True
                topic = response.choices[0].message.content
                wiki_topics.append(topic)

            except:
                wiki_topics.append("")
                error = sys.exc_info()[0]
                num_rate_errors += 1
                if error == BadRequestError: #OpenAIError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(
                        f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n"
                    )
                    assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))

    wiki_topics = [wiki_topic.replace("-", " ") for wiki_topic in wiki_topics]
    wiki_topics = [wiki_topic.replace("_", " ") for wiki_topic in wiki_topics]

    return wiki_topics


def csv_to_jsonl_for_factscore(results_dir):
    # get a list of all logs
    extension = "csv"
    runs = glob.glob("{}/*.{}".format(results_dir, extension))
    print(runs)
    jsonl_paths = []
    for run in runs:
        path_d = run.replace(".csv", ".jsonl")
        print(path_d)
        jsonl_paths.append(path_d)

        df = pd.read_csv(run, encoding="ISO-8859-1")
        #df = df.head(25)
        # df = df.loc[[1,2, 3,18, 22,23,5,30,65,70,18]]

        # add columns required by factscore
        df["topic"] = get_wiki_topic(df["prediction_sum"])
        # df["topic"] = search_wiki(df["prediction-summary"])
        df["cat"] = [[] for i in range(len(df))]

        # only keep the columns needed for factscore
        df = df[["article", "prediction_sum", "topic", "cat"]]
        df.columns = ["input", "output", "topic", "cat"]

        # convert each row to a dict and write as a jsonl
        df_jsonl = df.to_dict("records")
        with open(path_d, "w") as out:
            for ddict in df_jsonl:
                jout = json.dumps(ddict) + "\n"
                out.write(jout)

    return jsonl_paths


def df_to_jsonl_for_factscore(df, predictions_col_name):
    # extension = "csv"
    # runs = glob.glob("{}/*.{}".format(results_dir, extension))
    # print(runs)
    print("DF Head: \n", df.head())
    jsonl_paths = []
    # for run in runs:
    file_name = "json_data/{}.jsonl".format(predictions_col_name) #+".jsonl"
    # if ".csv" in file_name:
    #     file_name = file_name.replace(".csv", ".jsonl")
    # # else:
    # #     path_d = file_name + ".jsonl"
    print("Path to which JsonL will be stored: ", file_name)
    # jsonl_paths.append(path_d)

    # df = pd.read_csv(run, encoding="ISO-8859-1")
    # df = df.head(25)
    # df = df.loc[[1,2, 3,18, 22,23,5,30,65,70,18]]

    # add columns required by factscore
    df["topic"] = get_wiki_topic(df[predictions_col_name])
    # df["topic"] = search_wiki(df["prediction-summary"])
    df["cat"] = [[] for i in range(len(df))]

    # only keep the columns needed for factscore
    df = df[["article", predictions_col_name, "topic", "cat"]]
    df.columns = ["input", "output", "topic", "cat"]

    # convert each row to a dict and write as a jsonl
    df_jsonl = df.to_dict("records")
    with open(file_name, "w") as out:
        for ddict in df_jsonl:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

    return file_name # jsonl_paths
