from __future__ import print_function

import sys
from operator import add
import datetime
import time
import csv
import pandas as pd
import re
import numpy as np
import math

from numpy.linalg import inv
from scipy import stats
from pyspark.sql import SparkSession 
from pyspark.sql import SQLContext


# Find relative frequency for each of the 1000 words in each review
# Format: (line, map(word, relative_frequency))
def find_rel_freq(line):
  top_1000_words = top_1000_broadcast.value

  reg = re.compile('((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))')
  word_list = reg.findall(str(line[1]))

  line_count = len(word_list)
  d = {}

  if line_count != 0:
    for word in top_1000_words:
      word_count = word_list.count(word)
      rel_freq = word_count/line_count
      d[word] = rel_freq
    return (line[0], d)

# Standardize array
def standardize(x):
  x = np.array(x).astype(np.float)
  return (x - x.mean()) / x.std()

# Calculate p value for linear regression
def calculate_p_value_lin_reg(x, y, beta_array):
  k = np.mean(x)
  rss = 0

  for i, j in zip(x, y):
      rss = rss + np.square(j - beta_array[0][0] - (i*beta_array[1][0]))
  
  df = np.size(x)-2
  s = rss/df
  z = 0
  
  for i in x:
      z = z + np.square(i-k)

  denominator = s/z
  dem = math.sqrt(denominator)
  t = beta_array[1][0]/dem

  p = 2*(stats.t.pdf(t,df=df))

  return p

# Linear regression function for each word
# Output format: (Word, (Beta_0, Beta_1), P_value)
def linear_regression(word):

    # Get broadcast values
    hm_rel_freq = rel_freq_map_broadcast.value
    hm_rating_verified = rating_verified_map_broadcast.value
    hm_line_set = lines_set_broadcast.value

    # Create x (relative frequency) and y (rating) array
    x, y = [], []
    for line in hm_line_set:
        x.append(hm_rel_freq[line][word])
        y.append(hm_rating_verified[line][0])
        
    # Convert to array and standardize
    x = standardize(x)
    y = standardize(y)
    
    # Make matrices
    x_mat = np.asmatrix(np.c_[x, np.ones(np.size(x))])
    y_mat = np.asmatrix(y)

    try:

      # Calculate beta values
      x_trans_mat = np.transpose(x_mat)
      beta_mat = np.dot(np.dot(np.linalg.pinv(np.dot(x_trans_mat, x_mat)), x_trans_mat), np.transpose(y_mat))
      beta_array = np.asarray(beta_mat)

      # Multiplied by number of hypothesis (Bonferroni-correct p-value)
      p_value = calculate_p_value_lin_reg(x, y, beta_array) * 1000
      
      return (word, (beta_array[0][0], beta_array[1][0]), p_value)

    except np.linalg.LinAlgError as err:
      pass


def calculate_p_value_mult_lin_reg(x, x1, y, beta_array):
  rss = 0

  for i, j, k in zip(x, y, x1):
      rss = rss + np.square(j - beta_array[0][0] - (i*beta_array[1][0]) - (k*beta_array[2][0]))

  df = np.size(x)
  s = rss/df
  z = 0
  m = 0
  k = np.mean(x)
  v = np.mean(x1)
  
  for i in x:
      z = z + np.square(i-k)
  for j in x1:
      m = m + np.square(j-v)

  dem = math.sqrt(s/(z+m))
  t = beta_array[1][0]/dem
  t1 = beta_array[0][0]/dem

  p = 2*(stats.t.pdf(t1,df=df))

  return p


def multiple_linear_regression(word):
    
    # Get broadcast values
    hm_rel_freq = rel_freq_map_broadcast.value
    hm_rating_verified = rating_verified_map_broadcast.value
    hm_line_set = lines_set_broadcast.value

    # Create x (relative frequency), x1 (verified) and y (rating) array
    x, x1, y = [], [], []
    for line in hm_line_set:
        x.append(hm_rel_freq[line][word])
        x1.append(hm_rating_verified[line][1])
        y.append(hm_rating_verified[line][0])
        
    # Convert to array and standardize
    x = standardize(x)
    x1 = standardize(x1)
    y = standardize(y)

    # Make matrices
    mat_x = np.asmatrix(np.c_[np.asmatrix(np.c_[x, x1]), np.ones(np.size(x))])
    mat_y = np.asmatrix(y)

    try:
      
      # Calculate beta values
      x_trans_mat = np.transpose(mat_x)
      beta_mat = np.dot(np.dot(np.linalg.pinv(np.dot(x_trans_mat, mat_x)), x_trans_mat), np.transpose(mat_y))
      beta_array = np.asarray(beta_mat)

      # Multiplied by number of hypothesis (Bonferroni-correct p-value)
      p = calculate_p_value_mult_lin_reg(x, x1, y, beta_array) * 1000

      return (word, (beta_array[0][0], beta_array[1][0], beta_array[2][0]), p)
    
    except np.linalg.LinAlgError as err:
      pass


if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("Hypothesis Testing")\
        .getOrCreate()

    sqlContext = SQLContext(spark.sparkContext)

    # Specified regex to get words
    reg = re.compile('((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))')
    
    # Get lines of reviews
    lines = sqlContext.read.json(sys.argv[1]).select("reviewerID", "reviewText", "overall", "verified")
    
    # Find top 1000 words
    top_1000_words = lines.rdd.map(lambda row: row[1]) \
                      .flatMap(lambda word: reg.findall(str(word))) \
                      .map(lambda x: (x.lower(), 1)) \
                      .reduceByKey(add) \
                      .sortBy((lambda x: x[1]), False) \
                      .map(lambda x: x[0]) \
                      .take(1000)

    top_1000_broadcast = spark.sparkContext.broadcast(top_1000_words)

    # Find relative frequency of each word in each review
    # Format Map (line, map(word, rel_freq))
    # Filter the reviews which have no words in the review
    rel_freq_map = lines.rdd.map(lambda x: find_rel_freq(x)).filter(lambda x: x!=None).collectAsMap()
    rel_freq_map_broadcast = spark.sparkContext.broadcast(rel_freq_map)

    # Get ratings and verified values for all reviews
    # Format: Map (line, (rating, verified))
    rating_verified_map = lines.rdd.map(lambda x: (x[0], (x[2], x[3]))).collectAsMap()
    rating_verified_map_broadcast = spark.sparkContext.broadcast(rating_verified_map)

    # Get all reviewer id's (universal set)
    lines_set = set(rating_verified_map.keys())
    lines_set_broadcast = spark.sparkContext.broadcast(lines_set)

    # Get RDD of top 1000 words
    words_rdd = spark.sparkContext.parallelize(top_1000_words)

    # For each word, perform linear regression
    # Return format: (Word, (Beta_0, Beta_1), P_value)
    lin_reg = words_rdd.map(lambda x: linear_regression(x)).filter(lambda x: x!=None)
    
    # For each word, perform multiple linear regression with controlling variable = verified
    # Return format: (Word, (Beta_0, Beta_1, Beta_2), P_value)
    mult_lin_reg = words_rdd.map(lambda x: multiple_linear_regression(x)).filter(lambda x: x!=None)
    

    # Print top 20 values for following cases:

    print("Positive correlation for linear regression: ")
    print(lin_reg.takeOrdered(20, lambda x: -x[1][1]), '\n')
    
    print("Negative correlation for linear regression: ")
    print(lin_reg.takeOrdered(20, lambda x: x[1][1]), '\n')
    
    print("Positive correlation for multiple linear regression (Controlling variable = verified): ")
    print(mult_lin_reg.takeOrdered(20, lambda x: -x[1][1]), '\n')
    
    print("Negative correlation for multiple linear regression (Controlling variable = verified): ")
    print(mult_lin_reg.takeOrdered(20, lambda x: x[1][1]), '\n')

    spark.stop()