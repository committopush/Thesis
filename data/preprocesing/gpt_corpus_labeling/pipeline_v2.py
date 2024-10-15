#!/.conda

import pandas as pd
import openai
from dotenv import dotenv_values
import json
import os
import time
import numpy as np

os.chdir('/Users/benedikt/PycharmProjects/thesis_yufei/data/preprocesing/gpt_corpus_labeling')

class Sentiment_Classifier:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.client = openai.Client(api_key=dotenv_values(".env")['OPENAI_API_KEY'])
        self.classification_prompt = "This is a classification prompt"

    def classify_sentiment(self, text, prompt, retries=1):
        for attempt in range(retries):
            try:
                # Send each full text (no batching) to GPT-4
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt + "\n" + text  # Send the full input text
                        }
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    top_p=0.1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "json_object"  # Enforce JSON response format
                    }
                )

                response_text = response.choices[0].message.content
                return response_text
            except Exception as e:
                print(f"Error in classification (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2)  # Retry after 2 seconds
        print(f"Failed to classify after {retries} attempts.")
        return None  # Return None if retries fail


if __name__ == '__main__':
    # Load the data and the prompt
    data = pd.read_excel('/Users/benedikt/PycharmProjects/thesis_yufei/data/preprocesing/gpt_corpus_labeling/v1_500_sample.xlsx')
    prompt = open('/Users/benedikt/PycharmProjects/thesis_yufei/data/preprocesing/gpt_corpus_labeling/prompt.txt', 'r').read()

    # Instantiate the classifier with the data
    classifier = Sentiment_Classifier(data)

    # Initialize the sentiment columns in the DataFrame
    data["polarizing"] = 0
    data["populist"] = 0
    data["extremist"] = 0
    data["neutral"] = 0

    # Process each speech, send it to GPT, and update the DataFrame
    for i in range(len(data)):
        speech_id = data.loc[i, 'speech_id_long']
        text = data.loc[i, 'speech_content']

        # Send each speech content to GPT
        print(f"Processing speech {i + 1}: {speech_id}")
        result = classifier.classify_sentiment(text, prompt)

        # case where no response is received from GPT or the API call fails
        if result is None:
            print(f"Empty or invalid response for speech {speech_id}")
            # Default to neutral labels
            data.at[i, 'polarizing'] = 0
            data.at[i, 'populist'] = 0
            data.at[i, 'extremist'] = 0
            data.at[i, 'neutral'] = 0
            continue

        try:
            # Parse the JSON response from GPT
            result_dict = json.loads(result)["0"]

            # Update the DataFrame with the parsed result (using your original method)
            data.at[i, 'polarizing'] = result_dict['pol']
            data.at[i, 'populist'] = result_dict['pop']
            data.at[i, 'extremist'] = result_dict['ext']
            data.at[i, 'neutral'] = result_dict['neu']

        # Handling cases where a response was received, but it is not in a valid JSON format or missing the expected keys
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON Decode Error for speech ID {speech_id}: {e}")
            print(f"Problematic result: {result}")
            # Default to zero labels in case of error
            data.at[i, 'polarizing'] = 0
            data.at[i, 'populist'] = 0
            data.at[i, 'extremist'] = 0
            data.at[i, 'neutral'] = 0

    # Save results to Excel files
    num_files = 5
    names = ['silja', 'anna', 'toni', 'antoine', 'bene']

    # Split the DataFrame into approximately equal parts
    data_chunks = np.array_split(data, num_files)

    # Save each chunk into a separate Excel file
    for i, chunk in enumerate(data_chunks):
        chunk.to_excel(f'{names[i]}_manual_pre_labeling_1510.xlsx', index=False)
