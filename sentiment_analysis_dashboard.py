import os
import pandas as pd
import re
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv


def get_meta_title(text):
    return text.split("\n")[0]

def get_meta_desc(text):
  for i in (text.split("\n")):
    if "Description" in i:
      return i

def get_emotion_polarity(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the emtion dictionary with emotion and polarity score without any explation \
  and extra text.\
  #output format: {{
    "Happiness": "A state of well-being and contentment.",
    "Sadness": "A feeling of sorrow or unhappiness.",
    "Anger": "A strong feeling of displeasure or hostility.",
    "Fear": "An emotional response to a perceived threat or danger.",
    "Surprise": "A sudden feeling of astonishment or wonder.",
    "Disgust": "A strong feeling of dislike or disapproval.",
    "Joy": "A feeling of great pleasure and happiness.",
    "Guilt": "A feeling of responsibility or remorse for a perceived wrongdoing.",
    "Shame": "A painful feeling regarding one's own actions or behavior.",
    "Confusion": "A state of being perplexed or unclear in one's mind.",
    "Gratitude": "A feeling of thankfulness and appreciation.",
    "Regret": "A feeling of sorrow or disappointment over something that has happened.",
    "Relief": "A feeling of reassurance and relaxation after a distressing situation.",
    "Hope": "The expectation of a positive outcome or future event.",
    "Embarrassment": "A feeling of self-consciousness or shame due to a mistake or awkward situation.",
    "Contempt": "A feeling of disdain or lack of respect for someone or something.",
    "Love": "An intense feeling of deep affection.",
    "Hate": "A strong feeling of intense dislike or aversion.",
    "Frustration": "A feeling of being upset or annoyed due to being unable to achieve something.",
    "Excitement": "A feeling of great enthusiasm and eagerness." }} \
  here is the text: {text}
  """
  return prompt


def get_sentiment_polarity(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the sentiment dictionary with sentiment and polarity score without any explation \
  and extra text.\
  #output format: {{"positive":polarity score, "negative":polarity score,"neutral":polarity score }} \n
  here is the text: {text}
  """
  return prompt

def get_product_type(text):
  splits=text.split("Output:")
  if len(splits)>1:
    return splits[1]
  else:
    return splits[0]

groq_api_key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model="llama-3.1-70b-versatile",
    api_key=groq_api_key
)


st.title("Caribbean Text Sentiment Analysis System")

text=st.text_input("Input Text Here")

if st.button("Analyze")

    # Use a pipeline as a high-level helper
    
    pipe = pipeline("text-classification", model="mrarish320/caribbean_english_sentiment_fine_tuned_bert")
    label=pipe(text)["Label"]
    polarity=pipe(text)["score"]
    
    res=llm.invoke(get_sentiment_polarity(text)).content
    st.write(res)



# selection = st.sidebar.radio(
#     "Select",
#     ("Separte Title and Description", "Get Product Type"))
# file_name=st.file_uploader("Upload the File", type=["xlsx"])

# if file_name is not None:
#     df=pd.read_excel(file_name)


# # col1,col2,col3,col4,col5=st.columns(5)
# if selection=="Separte Title and Description":
#     # with col3:
#     if st.button("Submit"):
#         df["meta_title"]=df["Unnamed: 2"].apply(lambda x:get_meta_title(x))
#         df["meta_desc"]=df["Unnamed: 2"].apply(lambda x:get_meta_desc(x))

#         df["meta_title"]=df["meta_title"].apply(lambda x:re.sub("\*\*Meta Title:\*\*","",x))
#         df["meta_title"]=df["meta_title"].apply(lambda x:x.strip())



#         df["meta_desc"]=df["meta_desc"].apply(lambda x:re.sub("\*\*Meta Description:\*\*","",x))
#         df["meta_desc"]=df["meta_desc"].apply(lambda x:x.strip())
#         st.write(df)

# if selection=="Get Product Type":
#     if st.button("Submit"):
#         df["product type"]=df["Unnamed: 3"].apply(lambda x:get_product_type(x))
#         st.write(df)
