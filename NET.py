import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# # # 📂 1. VERİYİ YÜKLE VE TEMİZLE
# # df = pd.read_csv("netflixData.csv")

# # df = df[['Title', 'Director', 'Cast', 'Genres', 'Description']]
# # df.dropna(inplace=True)

# # df['Director'] = df['Director'].fillna('').str.lower()
# # df['Cast'] = df['Cast'].fillna('').str.lower()
# # df['Description'] = df['Description'].fillna('').str.lower()
# # df['Title'] = df['Title'].str.lower()
# # df['Genres'] = df['Genres'].str.lower()

# # df['combined_features'] = df['Genres'] + " " + df['Director'] + " " + df['Cast'] + " " + df['Description']

# # # 🔠 2. TF-IDF VEKTÖRLEŞTİRME
# # vectorizer = TfidfVectorizer(stop_words="english")
# # tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# # # 📏 3. COSINE SIMILARITY MATRİSİ HESAPLA
# # cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# # # 🔍 4. ÖNERİ FONKSİYONU
# # def recommend_movies(title, num_recommendations=5):
# #     title = title.lower()
# #     if title not in df['Title'].values:
# #         return "Unfortunately, no movie found."
    
# #     idx = df[df['Title'] == title].index[0]
# #     sim_scores = list(enumerate(cosine_sim[idx]))
# #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
# #     recommended_titles = [df.iloc[i[0]]['Title'] for i in sim_scores]
# #     return recommended_titles

# # # 🎬 5. KULLANICIYA ÖNERİ SUNMA
# # movie_name = input("Enter a movie/series name: ")
# # recommendations = recommend_movies(movie_name, 5)
# # print(f"'{movie_name}' Suggestions for: {recommendations}")

# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # 📂 1. VERİYİ YÜKLE VE TEMİZLE
# df = pd.read_csv("netflixData.csv")

# df = df[['Title', 'Director', 'Cast', 'Genres', 'Description']]
# df.dropna(inplace=True)

# df['Director'] = df['Director'].fillna('').str.lower().str.strip()
# df['Cast'] = df['Cast'].fillna('').str.lower().str.strip()
# df['Description'] = df['Description'].fillna('').str.lower().str.strip()
# df['Title'] = df['Title'].str.lower().str.strip()
# df['Genres'] = df['Genres'].str.lower().str.strip()

# # Gizli karakterleri kaldır
# for column in ['Title', 'Director', 'Cast', 'Genres', 'Description']:
#     df[column] = df[column].str.replace(r'\\n|\\t', '', regex=True)

# df['combined_features'] = df['Genres'] + " " + df['Director'] + " " + df['Cast'] + " " + df['Description']
# print(df[df['Title'].str.contains('100 things to do before high school', case=False, na=False)])


# # 🔠 2. TF-IDF VEKTÖRLEŞTİRME
# vectorizer = TfidfVectorizer(stop_words="english")
# tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# # 📏 3. COSINE SIMILARITY MATRİSİ HESAPLA
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# # 🔍 4. ÖNERİ FONKSİYONU
# def recommend_movies(title, num_recommendations=5):
#     title = title.strip().lower()
#     matching_titles = df[df['Title'] == title]
    
#     if matching_titles.empty:
#         return "Unfortunately, no movie found."
    
#     idx = matching_titles.index[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
#     recommended_titles = [df.iloc[i[0]]['Title'] for i in sim_scores]
#     return recommended_titles

# # 🎬 5. KULLANICIYA ÖNERİ SUNMA
# movie_name = input("Enter a movie/series name: ")
# recommendations = recommend_movies(movie_name, 5)
# print(f"'{movie_name}' Suggestions for: {recommendations}")


# VERİYİ YÜKLE VE TEMİZLE
df = pd.read_csv("netflixData.csv")
df = df[['Title', 'Director', 'Cast', 'Genres', 'Description']]
df.dropna(inplace=True)

# İndeksleri sıfırla
df.reset_index(drop=True, inplace=True)

df['Director'] = df['Director'].fillna('').str.lower().str.strip()
df['Cast'] = df['Cast'].fillna('').str.lower().str.strip()
df['Description'] = df['Description'].fillna('').str.lower().str.strip()
df['Title'] = df['Title'].str.lower().str.strip()
df['Genres'] = df['Genres'].str.lower().str.strip()

# Gizli karakterleri kaldır
for column in ['Title', 'Director', 'Cast', 'Genres', 'Description']:
    df[column] = df[column].str.replace(r'\\n|\\t', '', regex=True)

df['combined_features'] = df['Genres'] + " " + df['Director'] + " " + df['Cast'] + " " + df['Description']

# TF-IDF VEKTÖRLEŞTİRME
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# COSINE SIMILARITY MATRİSİ
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ÖNERİ FONKSİYONU
def recommend_movies(title, num_recommendations=5):
    title = title.strip().lower()
    matching_titles = df[df['Title'] == title]
    
    if matching_titles.empty:
        return "Unfortunately, no movie found."
    
    idx = matching_titles.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    
    recommended_titles = [df.iloc[i[0]]['Title'] for i in sim_scores]
    return recommended_titles
