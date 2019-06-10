import os
import pandas as pd
import random

df =  pd.read_csv("datasets/Apple/products.csv", usecols=["id"], dtype={'id': 'int32'})
array = df['id'].values

reviews = 50000
users = 10000
len_array = len(array)

file_handle = open("reviews.txt", 'w')

for i in range(reviews):
    idx = random.randrange(len_array)
    user = random.randrange(users)
    rating = random.randrange(0, 10)
    file_handle.write(str(user) + "," + str(array[idx]) + "," + str(rating) + "\n")

file_handle.close()
