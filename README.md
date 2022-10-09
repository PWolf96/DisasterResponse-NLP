# DisasterResponse-NLP

## Project Description

This project focuses on analyzing text data. It uses Natural Language Processing to analyze messages and categorize them based on the words used.
In this case, it analyzes text responses to a disaster and classifies them so that they are sent to the appropriate response group. 
The project refers to 2 datasets:
 - messages - the messages that were sent
 - categories - how those messages have been categorized
The process contains, cleaning the data, then creating a machine learning pipeline which is used to train a model to do the aforementioned classification.
In the end, a web app will be created where the model will be able to classify messages entered by the user.

### Process

#### Data Cleaning
The biggest challenge in the cleaning process was to combine the 2 datasets 'messages' and 'categories' without creating unnecessary columns/rows.
To do that messages and categories were cleaned separately. Initially, duplicate values have been removed from the datasets. Then categories has gone through
the following cleaning steps:
 - Split the column "categories" into separate values  
 - Rename the columns to represent their category name
 - Iterate through all values and keep only the numbers. Then convert them to integers
 - Adding the id column from the old dataset to the cleaned dataset
 
Then a new dataframe was created by merging the cleaned versions of 'messages' and 'categories'
Finally, the merged dataset was cleaned from columns and rows which contained only 0s as they were deemed as unnecessary for the classification

#### Saving the database
The database was saved into an SQL file


#### Train the classifier
In order to train the classifierm the following steps needed to be performed:
 - Load the data from the SQL file
 - Create 2 variables: X - containing all the messages; y - containing the dataframe of categories
 - Tokenize the messages - each message has been split into words which have been transformed into their root form in lower cases.
 - Tokens were created for each word and its respective role in the sentence. All tokens have been added to a list
 - A model pipeline has been built with the following elements:
  - CountVectorizer - creating the text documents into a matrix of token counts
  - TfIdfTransformer - converting the text documents intoa  matrix of tf-idf features
  - LogisticRegression - classifying the dataset using Logistic regression

#### Evaluating the model
The model has been ran and the following results have been obtained.

![image](https://user-images.githubusercontent.com/94782650/194724350-842a7a2f-fe4f-417d-8d18-45cca0470631.png)

#### Creating a web app
The whole model has then been pushed into a web app which allows the user to enter messages and see how the model will classify them

##### An image of the front page
![image](https://user-images.githubusercontent.com/94782650/194724465-526048b8-2706-4e65-ac62-4d6dd03d1b70.png)

##### An image of a classified message
![image](https://user-images.githubusercontent.com/94782650/194724520-ddafa80c-32b7-4515-a683-a0978c8b55af.png)



### Future work
- Add more visualizations to the web app.
- Based on the categories that the ML algorithm classifies text into, advise some organizations to connect to.
- Customize the design of the web app.
- Deploy the web app to a cloud service provider.
- Improve the efficiency of the code in the ETL and ML pipeline.
- Balance the dataset

### Technologies used
This analysis was done in **Python3** with the help of the packages **Pandas, sqlalchemy, Numpy, Sklearn and Nltk** 

## How to use the project

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
    - To run ML pipeline that trains classifier and saves
   
    ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`


## Credits
This project was created as part of Udacitys Data Science Nanodegree


