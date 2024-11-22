import pandas as pd
from transformers import BertTokenizer
import torch
import numpy as np
import os
import statistics

def preprocess_data(input_file, output_prefix, tokenizer, max_length=128):

    df = pd.read_excel(input_file)
    df = df.dropna(subset=['speech_content'])
    label_cols = ['polarizing', 'populist', 'extremist', 'neutral']
    
    required_columns = ['speech_content', 'speech_id_long'] + label_cols
    df = df[required_columns]
    
    # Convert labels to tensors
    labels = df[label_cols].values.astype(float)
    labels = torch.tensor(labels, dtype=torch.float)
    
    # Initialize lists to store lengths and speech IDs
    original_lengths = []
    truncated_lengths = []
    truncated_differences = []
    truncation_count = 0
    speech_ids = []  # List to store speech IDs
    
    # Tokenize texts and collect statistics
    input_ids_list = []
    attention_mask_list = []
    
    for index, row in df.iterrows():
        text = row['speech_content']
        speech_id = row['speech_id_long']
        
        # Tokenize without truncation to get original length
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            max_length=None
        )
        original_length = len(tokens)
        original_lengths.append(original_length)
        
        # Tokenize with truncation and padding
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])
        
        truncated_length = min(original_length, max_length)
        truncated_lengths.append(truncated_length)
        
        # Check if truncation occurred
        if original_length > max_length:
            truncation_count += 1
            truncated_differences.append(original_length - max_length)
        
        # Append speech_id to speech_ids list
        speech_ids.append(speech_id)
    
    # Convert lists to tensors
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_mask_list, dim=0)
    
    # Prepare output directory in the current working directory
    output_dir = os.path.join(os.getcwd(), 'tokenised_bert-base-german-cased')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed data
    torch.save({'input_ids': input_ids, 'attention_mask': attention_masks},
               os.path.join(output_dir, f'{output_prefix}_inputs.pt'))
    torch.save(labels, os.path.join(output_dir, f'{output_prefix}_labels.pt'))
    
    # Save speech_ids in a text file
    with open(os.path.join(output_dir, f'{output_prefix}_ids.txt'), 'w', encoding='utf-8') as f:
        for speech_id in speech_ids:
            f.write(f"{speech_id}\n")
    
    # Save label names for reference
    np.save(os.path.join(output_dir, f'{output_prefix}_label_names.npy'), label_cols)
    
    # Calculate statistics
    total_texts = len(df)
    truncation_rate = (truncation_count / total_texts) * 100
    if truncated_differences:
        median_truncation_length = statistics.median(truncated_differences)
    else:
        median_truncation_length = 0
    
    print(f'Preprocessing completed for {input_file}. Data saved with prefix {output_prefix}.')
    print(f'Total texts: {total_texts}')
    print(f'Number of texts truncated: {truncation_count} ({truncation_rate:.2f}%)')
    print(f'Median length of truncated parts: {median_truncation_length} tokens')

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    
    # Preprocess training data
    preprocess_data(
        input_file=os.path.join(data_folder, 'train.xlsx'),
        output_prefix='train',
        tokenizer=tokenizer
    )
    
    # Preprocess validation data
    preprocess_data(
        input_file=os.path.join(data_folder, 'val.xlsx'),
        output_prefix='val',
        tokenizer=tokenizer
    )
