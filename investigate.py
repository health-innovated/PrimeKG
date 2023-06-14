# source/inspriation: https://www.andrewvillazon.com/clinical-natural-language-processing-python/#processing-clinical-text-with-scispacy

import pandas as pd
DEF_NOT_FOUND_FEEDBACK = "DEFINITION NOT FOUND"

primekg = pd.read_csv('kg.csv', low_memory=False)
primekg.query('y_type=="exposure"|x_type=="exposure"').x_type.drop_duplicates()

# primekg.columns

# Index(['relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name',
#     'x_source', 'y_index', 'y_id', 'y_type', 'y_name', 'y_source'],
#     dtype='object')


lookhere = ["anatomy", "biological_process", "disease", "drug", "exposure"]
ignorehere = ["gene/protein", "molecular_function", "cellular_component", "pathway"]

filter = primekg[["x_type", "y_type"]].where(~primekg.x_type.isin(ignorehere) & ~primekg.y_type.isin(ignorehere)).notna().all(axis=1)
primekg.loc[filter].drop(columns=["x_id", "y_id"])

d1 = primekg[["x_type", "x_name"]].loc[filter].drop_duplicates()
d1.columns = ["type", "name"]

d2 = primekg[["y_type", "y_name"]].loc[filter].drop_duplicates()
d2.columns = ["type", "name"]

careabout = ["relation", "display_relation", "x_name", "x_type", "y_name", "y_type"]
disease_df = primekg.loc[filter & ((primekg.x_type == "disease") | (primekg.y_type == "disease"))][careabout]
disease_df = disease_df[disease_df.relation.isin(["indication", "contraindication"])]

disease_df.sort_values(["relation", "x_type", "x_name", "y_type", "y_name"]).to_csv("treatments.csv", index=False)

umls_lookup_csv = "datasets/data/umls/umls_lookup.csv"
umls_lookup_df = pd.read_csv(umls_lookup_csv, low_memory=False)

umls_desc_csv = "datasets/data/umls/umls_description.csv"
umls_desc_df = pd.read_csv(umls_desc_csv, low_memory=False)
umls_desc_df = umls_desc_df.loc[umls_desc_df.SUPPRESS == 'N'][["CUI", "DEF"]]

import spacy
import scispacy
from scispacy.linking import EntityLinker
# nlp = spacy.load("en_core_sci_sm")
nlp = spacy.load("en_ner_bc5cdr_md")
# nlp = spacy.load("en_ner_bionlp13cg_md")
nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})

# doc = nlp("10q22.3q23.3 microduplication syndrome")
# doc = nlp("An allograft was used to recreate the coracoacromial ligaments and then secured to decorticate with a bioabsorbable tenodesis screw and then to the clavicle.")
doc = nlp("Patient has headache and sore throat")

for entity in doc.ents:
    kb_entries = entity._.kb_ents
    best_matches = sorted(kb_entries, key=lambda x: x[1], reverse=True)
    match_unfound = True
    while best_matches and match_unfound:
        candidate_match = best_matches.pop(0)
        cui = candidate_match[0]
        match_score = candidate_match[1]
        query_result = umls_desc_df.loc[umls_desc_df.CUI == cui]["DEF"]
        if query_result.count():
            description = query_result.iloc[0]
            match_unfound = False
        else:
            description = DEF_NOT_FOUND_FEEDBACK
            continue
    print("========================================")
    clinical_dictionary_entry = f"{entity.text} = {description}"
    print(clinical_dictionary_entry)
    print("========================================")


# umls_desc_df.loc[umls_desc_df.CUI == cui]
# umls_lookup_df.loc[umls_lookup_df.cui == cui]





# umls_lookup_df.loc[umls_lookup_df.cui == cui]

# umls_desc_df[["CUI", "DEF"]]

# umls_desc_df.columns