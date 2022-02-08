import pandas as pd
from sklearn.model_selection import train_test_split
import re
from CatBoost_Class import CatBoost

df = pd.read_json('data.json')
# 'level', 'description', 'title'
#######################################################################################################################
# There's a description in Japanese here that probably shouldn't be there. I'll remove that row.
df.drop(99, inplace=True)
df = df.reset_index(drop=True)
#######################################################################################################################
titles = dict(df['title'].value_counts())
#print(titles, len(set(list(df['title']))))
# There are many job titles (211) so these don't seem useful as categorical features. I'll merge them with the description into one string-type feature
# merge text columns into one feature column
df_copy = df.copy()
df_copy['text'] = df_copy['title'] + ' ' + df_copy['description']
df_copy = df_copy.drop(columns=['description', 'title'])
#######################################################################################################################
# Feature engineering:
# Find how many texts have information about the number of years' experience required. This should be a good indicator of 'level' and a good learning feature:
num_years = []
for x in df_copy['text']:
    " ".join(x)
    x = x.replace('\n', " ")
    # regex to find number (plus optional '+'-sign, e.g. '3+ years') plus digit
    years = re.findall('([.\d+]+)\s*(?:years)', x)
    if years:
            num_years.append(years[0].replace("+", ""))
    else:
        num_years.append(0)
num_years = [int(x) for x in num_years]
num_years = [0 if item > 8 else item for item in num_years]
df_copy['years_required'] = num_years
#print('Number of rows with years-required information:', len(df_copy.loc[df_copy['years_required']!= 0]))
#######################################################################################################################
# Find rows where level = 'nan'. This will be the dev/hold-out set at the very end
df_hold = df_copy[df.isna().any(axis=1)]
#######################################################################################################################
# Remove dev/hold-out set values tocreate train/test set
tr_te_df = pd.concat([df_copy,df_hold]).drop_duplicates(keep=False)
#######################################################################################################################
# Get counts of target class ("level"), then weights
counts = list(tr_te_df['level'].value_counts())
weights = [round(1 - round(x /sum(counts), 2), 2) for x in counts]
#######################################################################################################################
# Create train/test sets
X = tr_te_df
y = tr_te_df['level'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
######################################################################################################################
# Create class instance and classify
cat = CatBoost(text_col_name='text', num_col_name='years_required', class_weights=weights)
cat.fit(X_train, y_train)
cat.get_accuracy(X_test, y_test)
preds = cat.get_preds(df_hold)
cat.return_new_json(dataframe=df, column_name='level', predictions=preds)



