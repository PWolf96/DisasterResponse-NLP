#Importing relevant libraries
from dataclasses import replace
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    A method to load the CSV files and return dataframe objects
    Args:
    Messages_filepath - the filepath of the messages csv
    categories_filepath - the filepath of the categories csv

    Ouputs
    messages - dataframe
    categories - dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    
    return messages, categories

def clean_data(messages, categories):  
    '''
    A method to clean the passed datasets and combine them into a single one. 
    Cleaning refers to:
        - Removal of duplicate values
        - Converting the categories inputs into separate columns with their respective values
        - Removing messages which are not associated with any categories
        - Combining both messages and categories into a single dataset to be used for training

    Args:
    messages - dataframe
    categories - dataframe

    Outputs:
    df - a combined cleaned dataframe 
    '''

    #Drop duplicate values based on the "id" column
    categories = categories.drop_duplicates(subset=(["id"]), inplace=False)
    messages = messages.drop_duplicates(subset=(["id"]), inplace=False)
        
    #Split the column "categories" into separate values 
    categoriesSplit = categories["categories"].str.split(";", expand=True)

    
    #Select the first row of the new dataframe
    row = categoriesSplit.iloc[0]

    category_colnames = []
    #Split each string and append the category name into a list
    for name in row:
        col = name.split("-")[0]
        category_colnames.append(col)
        
        
    #Rename the columns to represent their category name
    categoriesSplit.columns = category_colnames  
    
    #Iterate through all values and keep only the numbers. Then convert them to integers
    for column in categoriesSplit:
        categoriesSplit[column] = categoriesSplit[column].str[-1]
        categoriesSplit[column] = categoriesSplit[column].astype(int)
    
    #Adding the id column to categoriesSplit
    categoriesSplit["id"] = categories["id"]
    
    #Merge "messages" with "categoriesSplit" into one dataframe based on "id"
    df = pd.merge(messages, categoriesSplit, on=['id'])
    
    #drop any rows and columns which have only 0s
    #columns
    df = df.loc[:, (df != 0).any(axis=0)]
    #rows
    df = df[~(df.loc[:, 'related':'direct_report'] == 0).all(axis=1)]
    
    return df


def save_data(df, database_filename):
    '''
    A method to save the dataframe into a sql database

    Args:
    df - the cleaned dataframe
    database_filename - the name of the database

    Output:
    SQL database containing the dataframe values
    '''
    filenameDb = database_filename.split("/")[1]
    engine = create_engine('sqlite:///' + filenameDb)
    filename = filenameDb.split(".")[0]
    df.to_sql(filename, engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}\n  DATABASE: {}'
              .format(messages_filepath, categories_filepath, database_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
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