import pandas as pd
import os

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
During the paraphrasing process, the original sentence IDs were appended with '_p' to indicate paraphrased sentences. 
But some original sentences were paraphrased twice, resulting in some sentences having the same ID.
This script assigns unique IDs to each sentence, including paraphrased sentences.
First paraphrased sentences are identified by the '_p1' suffix, the second paraphrased sentence by '_p2'
'''

def assign_unique_ids(df):
    df = df.copy()

    # Function to check if a speech_id_long is for a paraphrased sentence
    def is_paraphrased(speech_id):
        return str(speech_id).endswith('_p')

    # Separate original and paraphrased sentences
    originals = df[~df['speech_id_long'].apply(is_paraphrased)]
    paraphrased = df[df['speech_id_long'].apply(is_paraphrased)]

    # Process paraphrased sentences
    if not paraphrased.empty:
        paraphrased['base_id'] = paraphrased['speech_id_long'].str.rstrip('_p')
        paraphrased['paraphrase_count'] = paraphrased.groupby('speech_id_long').cumcount() + 1
        paraphrased['speech_id_long'] = paraphrased.apply(
            lambda x: f"{x['base_id']}_p{x['paraphrase_count']}", axis=1
        )

        # Drop helper columns
        paraphrased.drop(columns=['base_id', 'paraphrase_count'], inplace=True)

    df_processed = pd.concat([originals, paraphrased], ignore_index=True)

    return df_processed

if __name__ == '__main__':
    
    # Determine the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the data folder relative to the script directory
    data_folder = os.path.join(script_dir, 'data')

    training_file = os.path.join(data_folder, 'training_labels_final_comments.xlsx')
    validation_file = os.path.join(data_folder, 'validation_labels_final_comments.xlsx')
    training_output = os.path.join(data_folder, 'train.xlsx')
    validation_output = os.path.join(data_folder, 'val.xlsx')

    # Debug: Print the file paths to verify they're correct
    print(f"Training file path: {training_file}")
    print(f"Validation file path: {validation_file}")

    # Columns to drop
    columns_to_drop = ['summe', 'source_file', 'Raus/Unstimmig', 'Kommentar']

    # Process training data
    print('Processing training data...')
    train_df = pd.read_excel(training_file)
    train_df_unique = assign_unique_ids(train_df)
    train_df_unique.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    train_df_unique.to_excel(training_output, index=False)
    print(f'Training data processed and saved to {training_output}')

    # Process validation data
    print('Processing validation data...')
    val_df = pd.read_excel(validation_file)
    val_df_unique = assign_unique_ids(val_df)
    val_df_unique.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    val_df_unique.to_excel(validation_output, index=False)
    print(f'Validation data processed and saved to {validation_output}')