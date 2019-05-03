# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:01:18 2019

@author: Shabbir Khan
"""

import numpy as np
import pandas as pd

data = pd.read_csv("movies_genres_en.csv", delimiter="\t")
def get_data(row):
    for c in data.columns:
        if row[c]==1:
            return c
data_genres = data.drop(['plot', 'title'], axis=1)
genre_list = list(data.apply(get_data, axis=1))
categories_list = list(data_genres.columns.values)
categories_list.append('plot_lang')
dframe1 = data.drop(categories_list ,axis=1)
title_list = dframe1[['title']].values.tolist()
plot_list = dframe1[['plot']].values.tolist()
dframe = pd.DataFrame()
dframe['title']=title_list
dframe['plots']=plot_list
dframe['genres']=genre_list

dframe.to_csv(path_or_buf='movie_genre_new.csv', sep='\t',header=True, encoding='utf8', index=False)