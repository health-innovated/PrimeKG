# sourcing requirements from https://platform.openai.com/docs/guides/fine-tuning

import pandas as pd, random, spacy, scispacy
from scispacy.linking import EntityLinker

NUM_SAMPLES = 10000

primekg = pd.read_csv('kg.csv', low_memory=False)
umls_pkg_df = primekg.loc[(primekg.x_type == "disease") | (primekg.y_type == "disease") ]
df = umls_pkg_df.iloc[random.sample(range(len(umls_pkg_df)), NUM_SAMPLES)] if NUM_SAMPLES else umls_pkg_df
# df.columns

def tuning_prompt_df(ser_a: pd.Series, ser_b: pd.Series):
    return pd.DataFrame.from_dict({"prompt": ser_a, "completion": ser_b})

tunings = pd.DataFrame()

# Relationships
prompt_ser = df.apply(lambda row: f"What is the relationship between {row.x_name} and {row.y_name}?", axis=1)
completion_ser = df.apply(lambda row: f" {row.x_name} is a(n) {row.relation.replace('_', ' ')} for {row.y_name}." + "#END", axis=1)
tunings = pd.concat([tunings, tuning_prompt_df(prompt_ser, completion_ser)])

# Types
prompt_ser = df.apply(lambda row: f"What is type of thing is {row.x_name}?", axis=1)
completion_ser = df.apply(lambda row: f" {row.x_name} is a type of {row.x_type}." + "#END", axis=1)
tunings = pd.concat([tunings, tuning_prompt_df(prompt_ser, completion_ser)])

prompt_ser = df.apply(lambda row: f"What is type of thing is {row.y_name}?", axis=1)
completion_ser = df.apply(lambda row: f" {row.y_name} is a type of {row.y_type}." + "#END", axis=1)
tunings = pd.concat([tunings, tuning_prompt_df(prompt_ser, completion_ser)])

tunings.drop_duplicates().to_json("prompt_tuning.jsonl", lines=True, orient="records")


nlp = spacy.load("en_core_sci_sm")
# nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})
nlp.max_length = round(56293997, -len("56293997")) * 10

def token_count(tuning_df=tunings):
    df = tuning_df.drop_duplicates()
    df["dummy"] = 1
    all_text_ser = df.groupby(by="dummy").agg(lambda x: ' '.join(x)).iloc[0]
    prompt_tokens = nlp(all_text_ser.prompt)
    completion_tokens = nlp(all_text_ser.completion)
    return {
        "prompt-tokens": len(prompt_tokens),
        "completion-tokens": len(completion_tokens),
        "total-tokens": len(prompt_tokens) + len(completion_tokens),
    }

# princing from https://openai.com/pricing#language-models , for fine-tuning
tuning_pricing = {
    "Ada": {
        "Training": 0.0004 / 1000,
        "Usage": 0.0016 / 1000
    },
    "Babbage": {
        "Training": 0.0006 / 1000,
        "Usage": 0.0024 / 1000
    },
    "Curie": {
        "Training": 0.0030 / 1000,
        "Usage": 0.0120 / 1000
    },
    "Davinci": {
        "Training": 0.0300 / 1000,
        "Usage": 0.1200 / 1000
    }
}

num_tokens = token_count(tunings)

model = "Curie"
pricing = tuning_pricing[model]["Training"]
price_to_tune = num_tokens["total-tokens"] * pricing
print(f"Price to tune over the '{model}' model (rate of {pricing * 1000} / 1K tokens), with {num_tokens}:", f"${price_to_tune:,.2f}", sep="\n")

# openai api fine_tunes.create -t prompt_tuning_prepared.jsonl -m curie

# openai api fine_tunes.follow -i ft-di9xd4HaK01UFmZXfeNkWGs7

# openai api completions.create -m 'ft-di9xd4HaK01UFmZXfeNkWGs7' -p 'How should I treat a patient with heartburn?'