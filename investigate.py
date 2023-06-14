import pandas as pd
DEF_NOT_FOUND_FEEDBACK = "DEFINITION NOT FOUND"

primekg = pd.read_csv('kg.csv', low_memory=False)
primekg.query('y_type=="exposure"|x_type=="exposure"').x_type.drop_duplicates()

primekg.columns

# primekg.loc[primekg.y_source.drop_duplicates() == "umls"]


Index(['relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name',
       'x_source', 'y_index', 'y_id', 'y_type', 'y_name', 'y_source'],
      dtype='object')

primekg[["relation", "x_type", "x_name", "y_type", "y_name"]].describe()

# for col in ["relation", "x_type", "x_name", "y_type", "y_name"]:
for col in ["relation", "x_type", "y_type"]:
    uniques = primekg[col].drop_duplicates().sort_values()
    print("===============")
    print(col)
    print("===============")
    print(uniques)
    print("count::::", uniques.count())


lookhere = ["anatomy", "biological_process", "disease", "drug", "exposure"]
ignorehere = ["gene/protein", "molecular_function", "cellular_component", "pathway"]
(
    primekg
    .where(primekg.x_type.isin(lookhere).notna() | primekg.y_type.isin(lookhere).notna())
    .where(~primekg.x_type.isin(ignorehere) & ~primekg.y_type.isin(ignorehere))
)

filter = primekg[["x_type", "y_type"]].where(~primekg.x_type.isin(ignorehere) & ~primekg.y_type.isin(ignorehere)).notna().all(axis=1)
primekg.loc[filter].drop(columns=["x_id", "y_id"])

primekg.loc[filter][["relation", "display_relation", "x_name", "x_type", "y_name", "y_type"]].drop_duplicates()

primekg.loc[primekg.display_relation == "parent-child"].relation.drop_duplicates()

primekg.display_relation.drop_duplicates() == "parent-child"


primekg.loc[filter][["relation", "display_relation", "x_name", "x_type", "y_name", "y_type"]].drop_duplicates()


primekg.relation.drop_duplicates()

careabout = ["relation", "display_relation", "x_name", "x_type", "y_name", "y_type"]
primekg.loc[filter & ((primekg.x_type == "disease") | (primekg.y_type == "disease"))][careabout]

pd.union([
    primekg[["x_type", "x_name"]].drop_duplicates(),
    primekg[["y_type", "y_name"]].drop_duplicates()
])

d1 = primekg[["x_type", "x_name"]].loc[filter].drop_duplicates()
d1.columns = ["type", "name"]

d2 = primekg[["y_type", "y_name"]].loc[filter].drop_duplicates()
d2.columns = ["type", "name"]

df = pd.concat([d1,d2]).drop_duplicates()
df[df.type == "disease"].name.sort_values().map(lambda s: s.lstrip('"').rstrip('"')).to_json("disease_names.json"), index=False)

disease_names_ser = df[df.type == "disease"].name.sort_values() #.map(lambda s: s.lstrip('"').rstrip('"')).map(lambda s: s.lstrip("'").rstrip("'"))

# import json, csv
# json.dumps(disease_names_ser.to_list())
disease_names_ser.to_csv("disease_names.csv", index=False)
disease_names_ser.to_json("disease_names.json", indent=2)

pdf = primekg.loc[filter & ((primekg.x_type == "disease") | (primekg.y_type == "disease"))][careabout]
pdf[pdf.relation.isin(["indication", "contraindication"])]


import spacy
# nlp = spacy.load("en_core_sci_sm")
nlp = spacy.load("en_ner_bc5cdr_md")
doc = nlp("10q22.3q23.3 microduplication syndrome")

[token for token in doc]

doc = nlp("An allograft was used to recreate the coracoacromial ligaments and then secured to decorticate with a bioabsorbable tenodesis screw and then to the clavicle.")

fmt_str = "{:<15}| {:<6}| {:<7}| {:<8}"
print(fmt_str.format("token", "pos", "label", "parent"))

for token in doc:
    print(fmt_str.format(token.text, token.pos_, token.ent_type_, token.head.text))


import scispacy
from scispacy.linking import EntityLinker
nlp = spacy.load("en_core_sci_sm")

nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})


# doc = nlp("Patient has prostate cancer with metastatic disease to his bladder.")
fmt_str = "{:<20}| {:<11}| {:<6}"
print(fmt_str.format("Entity", "Concept ID", "Score"))

entity = doc.ents[1]

for kb_entry in entity._.kb_ents:
    cui = kb_entry[0]
    match_score = kb_entry[1]

    print(fmt_str.format(entity.text, cui, match_score))


umls_lookup_csv = "datasets/data/umls/umls_lookup.csv"
umls_lookup_df = pd.read_csv(umls_lookup_csv, low_memory=False)

umls_desc_csv = "datasets/data/umls/umls_description.csv"
umls_desc_df = pd.read_csv(umls_desc_csv, low_memory=False)
umls_desc_df = umls_desc_df[["CUI", "DEF"]]

# MAX_ENTS = 5
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

    # fmt_str = "{:<20}| {:<11}| {:<6}| {:<50}"
    # print(fmt_str.format("Entity", "Concept ID", "Score", "Description"))
    # print(fmt_str.format(entity.text, cui, match_score, description))

    print("========================================")
    clinical_dictionary_entry = f"{entity.text} = {description}"
    print(clinical_dictionary_entry)
    print("========================================")


umls_desc_df.loc[umls_desc_df.CUI == cui]
umls_lookup_df.loc[umls_lookup_df.cui == cui]







umls_desc_df[["CUI", "DEF"]]

umls_desc_df.columns