import pandas as pd
from sklearn.model_selection import train_test_split
import re

def preprocess(df):
        #######################################################################################################################
        df_copy = df.copy()
        df_copy['text'] = df_copy['title'] + ' ' + df_copy['description']
        df_copy = df_copy.drop(columns=['description', 'title'])
        #######################################################################################################################
        # Find rows where level = 'nan'. This will be the dev/hold-out set at the very end
        df_hold = df_copy[df.isna().any(axis=1)]
        #######################################################################################################################
        # Remove dev/hold-out set values to create train/test set
        tr_te_df = pd.concat([df_copy,df_hold]).drop_duplicates(keep=False)
        #######################################################################################################################
        # Get counts of target class ("level"), then weights
        counts = list(tr_te_df['level'].value_counts())
        weights = [round(1 - round(x /sum(counts), 2), 2) for x in counts]
        #######################################################################################################################
        return df



