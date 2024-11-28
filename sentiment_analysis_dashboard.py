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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from langchain_community.document_loaders import TextLoader


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# List of stop words from NLTK
stop_words = set(stopwords.words('english'))


# Load environment variables
load_dotenv()

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


def get_text_metrics(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the following information in dictionary without any explation \
  and extra text.\

  1. **Number of Tokens:** Calculate the total number of tokens in the text.  
  2. **Readability Score:** Assess the readability of the text using common metrics like the Flesch Reading Ease score and/or any other suitable readability metric. Provide the score only.  
  3. **Quality Score:** Evaluate the overall quality of the text on a scale of 1 to 10, considering factors like grammar, vocabulary richness, and clarity. Provide reasons for your rating.  
  4. **Tone:** Identify the tone of the text (e.g., formal, conversational, persuasive, neutral) and explain your reasoning.  
  5. **Coherence:** Assess how coherent the text is in presenting its ideas and maintaining logical flow. Provide a coherence score on a scale of 1 to 10, with an explanation.  
  6. **Intentions:** Analyze the underlying intentions or purposes of the text (e.g., to inform, persuade, entertain, or a combination of these).  


  #output format: {{
    "Number of Tokens": "score",
    "Readability Score": "score",
    "Quality Score": "score",
    "Tone": "value",
    "Coherence": "value",
    "Intentions": "value",
    }} \
  here is the text: {text}
  """
  return prompt

groq_api_key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

llm=()

# llm = ChatGroq(
#   temperature=0,
#   model="llama-3.1-70b-versatile",
#   api_key=groq_api_key)


st.title("Caribbean Text Sentiment Analysis System")


file=st.file_uploader("Upload File",["csv","xlsx","pdf","txt"])

if file!=None:
  # st.write(file.name)
  file_name=file.name
  file_extension=file_name.split(".")[-1]

if st.button("upload"):
  if file_extension=="xlsx" or file_extension=="csv":
      column_name=st.text_input("Write column name which contains the Text")
      if file_extension=="xlsx":
          df=pd.read_excel(file)
          df_text="".join(df[column_name.strip()].values)
          with open("df_text.txt") as f:
              f.write(df_text)
      if file_extension=="csv":
          df=pd.read_csv(file)
          df_text="".join(df[column_name.strip()].values)
          with open("df_text.txt") as f:
              f.write(df_text)
  
      with open("df_text.txt") as f:
          f=f.read()
      loader = TextLoader(f)
      loader.load()
      # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      text_chunks = text_splitter.split_documents(data)
      # print the number of chunks obtained
      # len(text_chunks)
  
  if file_extension=="txt":
      loader = TextLoader(file)
      loader.load()
      # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      text_chunks = text_splitter.split_documents(data)
      # print the number of chunks obtained
      # len(text_chunks)
  
  if file_extension=="pdf":
      loader=PyPDFLoader(file)
      data = loader.load()
      # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      text_chunks = text_splitter.split_documents(data)
      # print the number of chunks obtained
      # len(text_chunks)


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
        wordcloud = WordCloud(width=800, height=665, background_color='white',stopwords=stop_words).generate(text)
        
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
        ner=llm.invoke(get_ner(text)).content
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


    col1,col2=st.columns(2)
    with col1:
        text_metrics=llm.invoke(get_text_metrics(text)).content
        text_metrics=ast.literal_eval(text_metrics)
        metrics_names=[]
        metrics_values=[]
        for i in ["Readability Score","Quality Score","Coherence"]:
          metrics_values.append(text_metrics[i])
          metrics_names.append(i)
            
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=metrics_names,y=metrics_values,hue=metrics_names,ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # plt.show()
        ax.set_title("Text Metrics")
        # Display in Streamlit
        st.pyplot(fig)

    with col2:
        st.subheader("Additional Text Metrics") 
        st.subheader("Tone")
        st.write(text_metrics["Tone"])
        st.subheader("Coherence Score")
        st.write(text_metrics["Coherence"])
        st.subheader("Coherence")
        st.write(text_metrics["Coherence"])



        
    














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
