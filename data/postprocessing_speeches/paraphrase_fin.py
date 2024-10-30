import pandas as pd
import openai
from tqdm import tqdm
import time
import os
import sys
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to paraphrase text
def paraphrase_text(text, rhetoric_category, num_paraphrases=1):
    paraphrased_texts = []
    for _ in range(num_paraphrases):
        try:
            # Define the prompt for paraphrasing
            prompt = f"Bitte paraphrasiere den folgenden Text so, dass der rhetorische Stil, der als {rhetoric_category} Rhetorik beschrieben wird, erhalten bleibt. Die Paraphrasierung soll weiterhin {rhetoric_category} Rhetorik widerspiegeln, kann jedoch die ursprüngliche Bedeutung kreativ abwandeln, eine stark abweichende Wortwahl nutzen, die Ideen neu formulieren und die Satzstrukturen deutlich verändern.\n\nUrsprünglicher Text:\n\"{text}\"\n\nParaphrasierter Text:"

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=500,
            )

            # Extract the generated text
            paraphrased = response['choices'][0]['message']['content'].strip()
            paraphrased_texts.append(paraphrased)
            # Removed time.sleep(1) to speed up processing
        except Exception as e:
            print(f"Error during paraphrasing: {e}")
            paraphrased_texts.append(None)
    return paraphrased_texts

def main():
    # Load datasets
    training_df = pd.read_pickle('/Users/benedikt/PycharmProjects/thesis_yufei/data/postprocessing_speeches/training_data.pkl')
    validation_df = pd.read_pickle('/Users/benedikt/PycharmProjects/thesis_yufei/data/postprocessing_speeches/validation_data.pkl')

    # Add dataset identifiers
    training_df['dataset'] = 'training'
    validation_df['dataset'] = 'validation'

    combined_df = pd.concat([training_df, validation_df], ignore_index=True)

    num_paraphrases = 2
    test_mode = False  # Set to False to run full processing

    # Restore progress if checkpoint exists
    checkpoint_file = 'paraphrase_checkpoint.pkl'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            paraphrased_data, start_index = pickle.load(f)
        print(f"Progress restored. Resuming from index {start_index}.")
    else:
        paraphrased_data = []
        start_index = 0

    if test_mode:
        # Test mode code (if needed)
        pass  # Replace with your test mode code if necessary
    else:
        sample_df = combined_df.reset_index(drop=True)
        total_samples = sample_df.shape[0]

        # Iterate over the DataFrame
        try:
            for index in tqdm(range(start_index, total_samples), total=total_samples - start_index, initial=start_index):
                row = sample_df.loc[index]
                text = row['speech_content']
                labels = row[['polarizing', 'populist', 'extremist', 'neutral']]
                rhetoric_categories = [label for label in labels.index if labels[label] == 1]
                rhetoric_category = ' and '.join(rhetoric_categories)

                paraphrased_texts = paraphrase_text(text, rhetoric_category, num_paraphrases=num_paraphrases)

                for paraphrased_text in paraphrased_texts:
                    if paraphrased_text:
                        new_row = row.copy()
                        new_row['speech_content'] = paraphrased_text
                        # Append '_p' to 'speech_id_long'
                        new_row['speech_id_long'] = f"{row['speech_id_long']}_p"
                        # Set 'source_file' to 'paraphrased'
                        new_row['source_file'] = 'paraphrased'
                        paraphrased_data.append(new_row)

                # Save progress after each iteration
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump((paraphrased_data, index + 1), f)

        except KeyboardInterrupt:
            # Save progress if manually interrupted
            with open(checkpoint_file, 'wb') as f:
                pickle.dump((paraphrased_data, index))
            print(f"\nProcessing interrupted at index {index}. Progress saved.")
            sys.exit(0)
        except Exception as e:
            # Save progress if an error occurs
            with open(checkpoint_file, 'wb') as f:
                pickle.dump((paraphrased_data, index))
            print(f"An error occurred: {e}. Progress saved up to index {index}.")
            sys.exit(1)

        # Remove checkpoint file after completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        # Save the augmented datasets
        paraphrased_df = pd.DataFrame(paraphrased_data)
        augmented_df = pd.concat([combined_df, paraphrased_df], ignore_index=True)

        # Ensure columns are in the desired order
        columns_order = ['speech_content', 'speech_id_long', 'polarizing', 'populist', 'extremist', 'neutral', 'source_file']
        augmented_df = augmented_df[columns_order + [col for col in augmented_df.columns if col not in columns_order]]

        # Remove 'dataset' column and save datasets
        augmented_training_df = augmented_df[augmented_df['dataset'] == 'training'].drop(columns=['dataset'])
        augmented_validation_df = augmented_df[augmented_df['dataset'] == 'validation'].drop(columns=['dataset'])

        augmented_training_df.to_pickle('augmented_training_data.pkl')
        augmented_validation_df.to_pickle('augmented_validation_data.pkl')

        print("Paraphrasing complete. Augmented datasets saved.")

if __name__ == "__main__":
    main()
