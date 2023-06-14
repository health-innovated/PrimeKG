import os
import numpy as np
import pandas as pd

# with open('../data/umls/MRCONSO.RRF', 'r') as f: 
umls_type = "lookup"
# umls_type = "description"
umls_docs = {
    "lookup": "MRCONSO",
    "description": "MRDEF",
}
umls_doc_base = umls_docs[umls_type]
with open(f'/Users/mreca/innovated/umls-2022AA/2022AA/META/{umls_doc_base}.RRF', 'r') as f: 
    data = f.readlines()
data = [[c for c in x.split('|') if c != '\n'] for x in data]
columns = {
    "lookup": ['cui', 'language', 'term_status', 'lui', 'string type', 'string_identifier', 'is_preferred', 'aui', 'source_aui', 'source_cui', 'source_descriptor_dui', 'source', 'source_term_type', 'source_code', 'source_name', 'x', 'x', 'x'],
    "description": ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF"]
}[umls_type]

df_umls = pd.DataFrame(data, columns = columns)
if umls_type == "lookup":
    df_umls = df_umls.query('language=="ENG"')
df_umls.head()

# df_umls.to_csv('../data/umls/umls.csv', index=False)
df_umls.to_csv(f'datasets/data/umls/umls_{umls_type}.csv', index=False)

# data[0]