# deep_learning_challenge
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
<h1>Report</h1>
The purpose of the analysis is to create a binary classifier that can predict whether the applicants will be successful if funded by Alphabet Soup. The goal is to try to choose applicants that will be successful so that funding is not wasted on an unsuccessful project.

- The target variable for the model is ‘IS_SUCCESSFUL’ column which is our y
- The features for the model is every column but the ‘IS_SUCCESSFUL’ column which is "APPLICATION_TYPE" , "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT”.
- The features that should be removed since they are not targets or features are: ‘EIN’ and ‘NAME’
- I originally used 2 hidden layers and an output layer. My first layer had 8 neurons, and relu activation algorithm. My second layer had 4 neurons and also relu activation. My output layer had 1 unit, with a sigmoid activation and an Adam optimizer when compiling.
- I was not able to achieve the target model performance of 75%. I used 4 different models and amounts of neurons, but was still unable to reach the threshold. All of my models finished with ~72% accuracy.
- In order to try to reach the threshold, I experimented with different amounts of neurons. First I increased the amount of neurons, but when I saw that I was still unable to reach the threshold, I decreased the neurons from my original model to see if that would allow me to reach the threshold. 
While the models were unable to reach the target model performance of 75%, they were still able to find patterns in the data and achieve a moderate accuracy. I think it would be interesting to experiment with other techniques to see what would be able to lead to a more accurate model. One model that would be interesting to explore would be a random forest model because of its simplicity, ability to handle noisy features, and ability to handle high dimensional data. 
