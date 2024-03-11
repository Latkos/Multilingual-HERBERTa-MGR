import pandas as pd
import re

def caesar_cipher(text, shift):
    def cipher_letter(letter):
        if 'a' <= letter <= 'z':
            return chr(((ord(letter) - ord('a') + shift) % 26) + ord('a'))
        elif 'A' <= letter <= 'Z':
            return chr(((ord(letter) - ord('A') + shift) % 26) + ord('A'))
        else:
            return letter

    return ''.join(cipher_letter(char) for char in text)

def shift_text_except_tags(text, shift=1):
    # Regular expression to detect tags and avoid changing them
    tags_pattern = re.compile(r"(<e[12]>|<\/e[12]>)")
    parts = tags_pattern.split(text)  # Split the text by tags
    new_parts = [part if tags_pattern.match(part) else caesar_cipher(part, shift) for part in parts]
    return ''.join(new_parts)

def apply_caesar_cipher_to_dataframe(file_path,save_path=None, save=True):
    if save_path==None:
        save_path='../data_augmented/{file_path}_caesar.tsv'
    df=pd.read_csv(file_path,sep='\t')
    df['entity_1'] = df['entity_1'].apply(lambda x: shift_text_except_tags(x, shift=1))
    df['entity_2'] = df['entity_2'].apply(lambda x: shift_text_except_tags(x, shift=1))
    df['text'] = df['text'].apply(lambda x: shift_text_except_tags(x, shift=1))
    if save:
        df.to_csv(save_path,index=False,sep='\t')
    return df

