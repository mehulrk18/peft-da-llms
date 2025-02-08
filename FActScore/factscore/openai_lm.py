import logging
import os
import sys
import time

import numpy as np
import openai
from openai import OpenAI, BadRequestError # OpenAIError

# from metrics.FActScore.factscore.lm import LM
from FActScore.factscore.lm import LM


class OpenAIModel(LM):
    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.client = None  # Initialized with load_model() method
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key.strip())
        # next two lines for replacing above commented line
        # openai.api_key = api_key.strip()
        # self.client = openai
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=16384, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(
                message, self.client, temp=self.temp, max_len=max_sequence_length
            )
            # Get the output from the response
            output = response.choices[0].message.content.strip()
            return output, response
        elif self.model_name == "gpt-4o-mini":
            # Call API
            response = call_GPT4(prompt, self.client, temp=self.temp)
            # Get the output from the response
            output = response.choices[0].message.content
            return output, response
        else:
            raise NotImplementedError()


def call_ChatGPT(
    message,
    openai_client,
    model_name="gpt-4o-mini",
    max_len=16384,
    temp=0.7,
    verbose=False,
):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": message},
                ],
                max_tokens=max_len,
                temperature=temp,
            )
            received = True
        except:
            # print(message)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == BadRequestError: # OpenAIError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"BadRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False

            logging.error(
                "API error: %s (%d). Waiting %dsec"
                % (error, num_rate_errors, np.power(2, num_rate_errors))
            )
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT4(
    prompt,
    openai_client,
    model_name="gpt-4o-mini",
    max_len=16384,
    temp=0.7,
    num_log_probs=0,
):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_len,
                temperature=temp,
                logprobs=True,  # Needs to be True if num_log_probs > 0
                top_logprobs=num_log_probs,
            )
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == BadRequestError: #OpenAIError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
