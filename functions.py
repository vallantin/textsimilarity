import pandas as pd
import ast, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import streamlit as st

@st.cache()
def preprocess(text):

    text = str(text)

    # lower the text
    text = text.lower()

    # remove numbers
    text = re.sub(r'\d+', '', text)
    
    return text

@st.cache(allow_output_mutation=True)
def process_vectors(vectors_df):
    
    vectors_df['vector'] = vectors_df['vector'].map(lambda x: x.replace('[[','').replace('[',''))
    vectors_df['vector'] = vectors_df['vector'].map(lambda x: x.replace(']]','').replace(']',''))
    vectors_df['vector'] = vectors_df['vector'].map(ast.literal_eval)
    vectors_df['vector'] = vectors_df['vector'].map(np.array)

    return vectors_df

# load bert model
# --------------------------------------------------
# Generating Embeddings using the sentence_transformer 
# library to encode the sentence into vectors.
@st.cache()
def load_bert_model():
    try:
        return SentenceTransformer('bert-base-uncased')
    except:
        return None

@st.cache()
def preprocess_text(text):

    # remove text between []
    text = re.sub(r"(\[.*\])+",'', text)
    text = re.sub(r"/s+",'/s', text)
    text = re.sub(r"^-/s",'', text)
    return text

@st.cache()
def load_data():
    data = pd.read_csv('data/data.csv')
    data['title'] = data['title'].map(preprocess_text)
    data = data[data['title'] != '']
    data.sort_values('title', inplace=True)
    data.reset_index(inplace=True)
    return data

#---------------------------------------------------------------------------------------
# get the data
def get_snippets(KEYWORD, N_RESULTS=5):

    # load the data
    data = load_data()

    # load the vectors dataframe
    vectors_df = data[['title', 'vector']]

    # convert the vectors to numpy arrays
    vectors_df = process_vectors(vectors_df)

    # keep only title
    data = data[['title']]
       
    # get the vector for the keyword
    query_embeddings = [vectors_df[vectors_df.title == KEYWORD]['vector'].to_list()[0]]

    # calculate the distance between the embedding and the vectors
    def get_distance(embedding, vector):
        return np.linalg.norm(embedding - vector)
            
    results_df = vectors_df.copy()
    results_df['distances'] = results_df['vector'].apply(lambda vector: get_distance(query_embeddings, vector))
            
    # sort the distances from the smallest to the largest
    results_df = results_df.sort_values(by='distances')

    # present the most similar discussions.
    # remove any discussion if the id is exactly the same as the row
    results_df = results_df[['title','distances']]
    results_df = results_df.head(N_RESULTS)

    # since the text used is already in the dataset, then we have to remove it from the results
    # because the distance to itself will be 0
    results_df = results_df[results_df['title'] != KEYWORD]

    return results_df