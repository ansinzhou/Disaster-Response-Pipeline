# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Required libraries:
 - nltk
 - numpy
 - pandas
 - scikit-learn
 - sqlalchemy
 
### Motivation:
  The project aims to analyze data from Figure Eight and build a model to provide classification for disaster messages.
  
  The project includes a flask web app where a disaster message can be entered , and will be classified into one or several emergency categories. The wb app also displays visualization of data.
  
### Acknowldegement:
  I would like to thank Figure Eight for the data used in the project, and Udacity for the advice and review.
