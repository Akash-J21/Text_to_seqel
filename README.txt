Text to sequel or sql is a NLP based ML program developed using python language

The code is developed in pycharm community environment.
The API used here is POSTMAN to GET and POST  
Note: Only text input is acceptable 

Using this text_to_sql code, user can translate the human language input into machine understanding language. Here the relational database is used to store the data that to be fetched, so we tranforms the input code to sql or sequel query.

Step 1 : Load the data (stock_file_csv) into the database  (Here Microsoft Seqel Server Database is used) in the name of stock_file

Step 2 : Copy the python code and run it

Step 3 : Paste the host url and copy it into the API (Postman)

Step 4 : Use GET method to test the connectivity between the API and python

Step 5 : Once the connectivity is valid, Use the POST method
	Set the Key as Question
	Set the Value as the question that user wants to fetch from the database

Step 6 : Send the connection and backend process fethces the required data from the database.
