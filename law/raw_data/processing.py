import re
from tqdm import tqdm
import pandas as pd
import json
import openai

openai.api_key = "..."
MODEL_NAME = "gpt-4o-mini" 

def openai_prompt(message_history, text_only=True):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=message_history,
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if text_only:
            return response.choices[0].message.content.strip()
        else:
            return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_entities_and_info(text_snippet):
    user_prompt = f"""
You are a helpful assistant that extracts specific data from legal text.

Given the following text, do the following:

1) Extract all of the criminal behavior or convictions associated with the defendant.
   - These should be single words or short phrases.

2) Extract other identifiable information about the defendant.
   - These should be single words or 2-3 word phrases that identify the defendant (e.g., name, address, nickname, etc.) or potentially relevant personal details (e.g., "owns two dogs").

**IMPORTANT**: 
- Output your results in a JSON-like format, wrapped exactly and only between <Answer> and </Answer> tags.
- Use the following format:

<Answer>
{{
  "criminal_behaviors": ["list", "of", "criminal", "behaviors"],
  "identifiable_info": ["list", "of", "other", "identifiable", "info"]
}}
</Answer>

Here is the text to analyze:
{text_snippet}
    """

    message_history = [
        {
            "role": "system",
            "content": "You are a data extraction assistant that follows user instructions carefully."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = openai_prompt(message_history)

    if not response:
        return None

    answer_match = re.search(r"<Answer>(.*?)</Answer>", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    else:
        return None
    
df = pd.read_csv("possession_legal_data.csv")

df_to_parse = df[df["Nomic Topic: medium"] == "Sentencing"]

# add column to df_to_parse to store extracted answer
df_to_parse["criminal_behaviors"] = ""
df_to_parse["identifiable_info"] = ""
for i, row in tqdm(df_to_parse.iterrows(), total=df_to_parse.shape[0]):
    extracted_answer = extract_entities_and_info(row["text"])
    if extracted_answer:
        data = json.loads(extracted_answer)
        df_to_parse.at[i, "criminal_behaviors"] = data.get("criminal_behaviors", "")
        df_to_parse.at[i, "identifiable_info"] = data.get("identifiable_info", "")

df_to_parse.to_csv("sentencing_data_with_extraction.csv", index=False)

def get_short_summary(text_snippet):
    system_prompt = (
        "You are a helpful assistant that summarizes legal text. "
        "Your summaries are concise and focus on the main points."
    )

    user_prompt = f"""
Please summarize the following legal text in a short paragraph (1-3 sentences max):

{text_snippet}
"""

    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = openai_prompt(message_history)
    print(response)
    return response

df_to_parse_with_summary = df_to_parse.copy()

for i, row in tqdm(df_to_parse.iterrows(), total=df_to_parse.shape[0]):
    df_to_parse.at[i, "summary"] = get_short_summary(row["text"])

df_to_parse_with_summary.to_csv("sentencing_data_with_summary.csv", index=False)