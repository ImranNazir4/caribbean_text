import os
import pandas as pd
import re
import ast
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# List of stop words from NLTK
stop_words = set(stopwords.words('english'))


# Load environment variables
load_dotenv()

def get_meta_title(text):
    return text.split("\n")[0]

def get_meta_desc(text):
  for i in (text.split("\n")):
    if "Description" in i:
      return i

def get_ner(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the NER if there any in the text in form of dictionary without any explation \
  and extra text.\
  #Entities to target: PPERSON, LOCATION, ORGANIZATION, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL \
  #output format: {{"word":"entity_label"}}
  here is the text: {text}
  """
  return prompt



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

if st.button("Analyze"):

    # Use a pipeline as a high-level helper
    
    pipe = pipeline("text-classification", model="mrarish320/caribbean_english_sentiment_fine_tuned_bert")
    # label=pipe(text)[0]["label"]
    # polarity=pipe(text)[0]["score"]

    # st.write(polarity)
    sentiment=llm.invoke(get_sentiment_polarity(text)).content
    sentiment=ast.literal_eval(sentiment)
    # st.write(res)

    emotion=llm.invoke(get_emotion_polarity(text)).content
    emotion=ast.literal_eval(emotion)

    col1,col2=st.columns(2)

    with col1:
        # Create a Seaborn bar plot
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=sentiment.keys(), y=sentiment.values(),hue=sentiment.keys(),ax=ax)
        plt.title("Sentiment Analysis")
        # plt.xticks(rotation=90)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # plt.show()
        ax.set_title("Sentiment Analysis")
        # Display in Streamlit
        st.pyplot(fig)
   
    with col2:
        # Create a Seaborn bar plot
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=emotion.keys(), y=emotion.values(),hue=emotion.keys(),ax=ax)
        plt.title("Emotion Analysis")
        # plt.xticks(rotation=90)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # plt.show()
        ax.set_title("Emotion Analysis")
        # Display in Streamlit
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=700, background_color='white',stopwords=stop_words).generate(text)
        
        # Display word cloud using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")  # Hide axes
        ax.set_title("Word Cloud Visualization", fontsize=16, color="blue")
        st.pyplot(fig)
    
    with col2:

# Create a Seaborn bar plot
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5))
        # sns.barplot(x=emotion.keys(), y=emotion.values(),hue=emotion.keys(),ax=ax)
        # plt.title("Emotion Analysis")
        # plt.xticks(rotation=90)
        ner=llm.invoke(get_ner(caribbean_story)).content
        ner=ast.literal_eval(ner)
    
        # Convert the ner dictionary to a Pandas DataFrame for long-form data
        ner_df = pd.DataFrame({'entity_label': list(ner.values())})
        
        # Now, use the 'entity_label' column for both x and hue
        sns.countplot(x='entity_label', hue='entity_label', data=ner_df,ax=ax)
 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # plt.show()
        ax.set_title("NER Analysis")
        # Display in Streamlit
        st.pyplot(fig)

        
    














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
