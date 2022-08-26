from functions import *

# config layout
st.set_page_config(page_title='Document similarity',
                   #layout = 'wide',
                   page_icon = 'img/icon.png',
                   initial_sidebar_state = 'collapsed'
                   )

# hold missing questions
missing_questions = st.sidebar.empty()

st.header('🧬 Document similarity')

st.write("""
          Did you know that it is possible to measure the similarity of two or more documents using Natural Language Processing? 
          Through text vectors, we can check the distances between texts to verify whether they are semantically close. 
          This dashboard uses a dataset of reddit entries related to Machine Learning, Data Science and Tech. 
          Choose an entry from the list below and see which documents are semantically close to your choice.
        """)
     
#load the model and the data
info = st.empty()
info.info('⏳ Loading data: it may take a few seconds. Please, wait...')
data  = load_data()
info.empty()

# ------------------------------------------------------------------------------------
st.markdown("""---""")
st.subheader('Select one entry below')

# add an empty option
placeholder = 'Select one text from the list'
options = [placeholder] + data['title'].to_list()
text = st.selectbox("Select one entry", options = options)

if text != placeholder:
    # get the entities after prediction
    snippets = get_snippets(text, N_RESULTS=7)

    st.subheader('Similar texts:')
    st.table(snippets)
    
# ------------------------------------------------------------------------------------
st.markdown("""---""")
st.subheader('To go further')
st.write("[A Gentle Introduction to Vector Space Models](https://machinelearningmastery.com/a-gentle-introduction-to-vector-space-models/)")
st.write("[Measuring Text Similarity Using BERT](https://www.analyticsvidhya.com/blog/2021/05/measuring-text-similarity-using-bert/)")

st.subheader('Other projects')
st.write("[Job description generator](https://vallantin-jobdescriptiongenerator-app-5wz0u4.streamlitapp.com/)")
st.write("[NLP Resources dashboard](https://vallantin-nlp-resources-app-1c6nvk.streamlitapp.com/)")

# ------------------------------------------------------------------------------------
st.markdown("""---""")
st.image('img/icon.png', width=80)
st.markdown("<h6 style='text-align: left; color: grey;'>Made by <a href='https://wila.me/' target='_blank'>wila.me</a></h6>", unsafe_allow_html=True)



















