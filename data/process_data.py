import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=["id"])
    return df

def clean_data(df):
    df.drop('original', axis=1, inplace=True)
    # create a dataframe of individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the caregories dataframe
    row = categories.iloc[0]
    #use row to extract a list of new column names for categories              
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = categories[column].astype(int)
    # replace categories column in df with new category columns   
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis=1)
    # replace duplicates
    df = df.drop_duplicates(keep = "first")
   
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False)
    
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()