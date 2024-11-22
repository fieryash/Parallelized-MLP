import os
from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import IntegerType

os.environ["PYSPARK_PYTHON"] = r"C:\Users\ashtik\miniconda3\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\ashtik\miniconda3\python.exe"

def edit_distance(pair):
    str1, str2 = pair
    # Create a table to store results of subproblems
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Fill dp table
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # Min operations = j
            elif j == 0:
                dp[i][j] = i  # Min operations = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],    # Insert
                                   dp[i - 1][j],    # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]

@pandas_udf(IntegerType())
def compute_edit_distance_udf(str1_series, str2_series):
    results = []
    for str1, str2 in zip(str1_series, str2_series):
        results.append(edit_distance((str1, str2)))
    return pd.Series(results)


# Function to compute edit distances using multiple processes
def compute_edit_distance_multiprocess(pair, num_workers):
    # implement a multi-process version of edit_distance function
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(edit_distance, pair_data)
    return results


if __name__=="__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, default='simple-wiki-unique-has-end-punct-sentences.csv', help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, default=300, help="Number of sentences")
    args = parser.parse_args()

    # Number of processes to use (set to the number of CPU cores)
    num_workers = multiprocessing.cpu_count()
    # print(f'number of available cpu cores: {num_workers}')
    # Sample list of string pairs
    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Initialize Spark session
    # Convert pair data to Spark DataFrame
    # Register the edit distance function as a pandas UDF for single pairs
    # Compute edit distances using Spark
    spark = SparkSession.builder \
        .appName("EditDistance") \
        .getOrCreate()

    pair_df = pd.DataFrame(pair_data, columns=['str1', 'str2'])
    spark_df = spark.createDataFrame(pair_df)   

    start_1 = time.time()
    result_df = spark_df.withColumn("edit_distance", compute_edit_distance_udf(col("str1"), col("str2")))
    # result_df.show()
    end_1 = time.time()
    time_1 = end_1 - start_1
    # print(f"Time taken (Spark): {start_1 - end_1:.2f} seconds")

    # Stop the Spark session
    spark.stop()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Multi-process
    start_2 = time.time()
    # Compute edit distances in parallel
    edit_distances = compute_edit_distance_multiprocess(pair_data, num_workers)
    end_2 = time.time()
    time_2 = end_2 - start_2
    # print(f"Number of string pairs processed: {len(edit_distances)}")
    # print(f"Time taken (multi-process): {start_2 - end_2:.3f} seconds")


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Vanilla for loop
    start_3 = time.time()
    distances = []
    for pair in tqdm(pair_data, ncols=100):
        distances.append(edit_distance(pair))
    end_3 = time.time()
    time_3 = end_3 - start_3
    # print(f"Edit Distances (Non-Spark): {distances}")
    # print(f"Time taken (for-loop): {time_3:.3f} seconds")

    print(f"Time cost (Spark, multi-process, for-loop): [{time_1:.3f}, {time_2:.3f}, {time_3:.3f}]")