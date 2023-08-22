# App to predict penguin species
# Using user provided data for model training

# Import libraries
import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split


st.title('Penguin Classifier') 

st.write("This app uses 6 inputs to predict the species of penguin using " 

         "a model built on the Palmer's Penguin's dataset. Use the form below" 

         " to get started!") 

# Asking users to input their own data
penguin_file = st.file_uploader('Upload your own penguin data') 

# Setting the default option to load our Random Forest model
# if there is no penguin file
if penguin_file is None: 
    # Load default dataset
    # This dataset will be used later for plotting histograms
    # in case if the user does not provide any data
    penguin_df = pd.read_csv('penguins.csv') 
    rf_pickle = open('random_forest_penguin.pickle', 'rb') 
    map_pickle = open('output_penguin.pickle', 'rb') 
    rfc = pickle.load(rf_pickle) 
    unique_penguin_mapping = pickle.load(map_pickle) 
    rf_pickle.close() 
    map_pickle.close() 

# If the file is provided, we need to clean it and train a model on it
# similar to what we did in the Jupyter notebook
else: 
    # Load dataset as dataframe
    penguin_df = pd.read_csv(penguin_file) 
    # Dropping null values
    penguin_df = penguin_df.dropna() 
    # Output column for prediction
    output = penguin_df['species'] 
    # Input features (excluding year column)
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                           'flipper_length_mm', 'body_mass_g', 'sex']] 
    # One-hot-encoding for categorical variables
    features = pd.get_dummies(features) 
    # Factorize output feature (convert from string to number)
    output, unique_penguin_mapping = pd.factorize(output) 
    # Data partitioning into training and testing
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = .8) 
    # Defining prediction model
    rfc = RandomForestClassifier(random_state=15) 
    # Fitting model on training data
    rfc.fit(x_train, y_train) 
    # Making predictions on test set
    y_pred = rfc.predict(x_test) 
    # Calculating accuracy of the model on test set
    score = round(accuracy_score(y_pred, y_test), 2) 
    st.write('We trained a Random Forest model on these data,' 
             ' it has a score of {}! Use the ' 
             'inputs below to try out the model.'.format(score))

# After creating the model, we need inputs from the user for prediction
# NOTE: This time we make an improvement. In the previous case, each time
# a user changes an input in the app, the entire app reruns.
# We can use st.form() and st.submit_form_button() to wrap the rest of 
# user inputs in and allow the user to change all of the inputs and submit
# the entire form at once instead of mutliple times
with st.form('user_inputs'): 
  island = st.selectbox('Penguin Island', options=[
    'Biscoe', 'Dream', 'Torgerson']) 
  sex = st.selectbox('Sex', options=[
    'Female', 'Male']) 
  bill_length = st.number_input(
    'Bill Length (mm)', min_value=0) 
  bill_depth = st.number_input(
    'Bill Depth (mm)', min_value=0) 
  flipper_length = st.number_input(
    'Flipper Length (mm)', min_value=0) 
  body_mass = st.number_input(
    'Body Mass (g)', min_value=0) 
  st.form_submit_button() 

# Putting sex and island variables into the correct format
island_biscoe, island_dream, island_torgerson = 0, 0, 0 
if island == 'Biscoe': 
  island_biscoe = 1 
elif island == 'Dream': 
  island_dream = 1 
elif island == 'Torgerson': 
  island_torgerson = 1 

sex_female, sex_male = 0, 0 
if sex == 'Female': 
  sex_female = 1 
elif sex == 'Male': 
  sex_male = 1 

# Create prediction and display it to user
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, 
  body_mass, island_biscoe, island_dream, 
  island_torgerson, sex_female, sex_male]]) 
prediction_species = unique_penguin_mapping[new_prediction][0]
st.subheader("Predicting Your Penguin's Species")
st.write('We predict your penguin is of the {} species'.format(prediction_species)) 

# Showing Feature Importance plot
st.write('We used a machine learning model (Random Forest) to '
         'predict the species, the features used in this prediction '
         'are ranked by relative importance below.')
st.image('feature_importance.png')

# Adding histograms for continuous variables for model explanation
st.write('Below are the histograms for each continuous variable '
         'separated by penguin species. The vertical line '
         'represents your inputted value.')

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['bill_length_mm'], hue = penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['bill_depth_mm'], hue = penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['flipper_length_mm'], hue = penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Bill Flipper by Species')
st.pyplot(ax)