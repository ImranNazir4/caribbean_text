import os
import pandas as pd
import re
import streamlit as st




def get_meta_title(text):
    return text.split("\n")[0]

def get_meta_desc(text):
  for i in (text.split("\n")):
    if "Description" in i:
      return i


def get_product_type(text):
  splits=text.split("Output:")
  if len(splits)>1:
    return splits[1]
  else:
    return splits[0]

st.title("Caribbean Text Sentiment Analysis System")

text=st.text_input("Input Text Here")






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