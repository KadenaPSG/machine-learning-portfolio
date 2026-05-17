from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

application = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML Model Code --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/')
@application.route('/about')
def about():
    return render_template("about.html")

@application.route('/titanicPredictor')
def titanicPredictor():
    return render_template("titanicPredictor.html")

def preprocessDataAndPredict(age, sex, pclass):
    # Convert age to float
    age = float(age)

    # Define and instantiate the variables for the encoded columns
    sex_female = 0
    sex_male = 0
    pclass_1 = 0
    pclass_2 = 0
    pclass_3 = 0

    if sex == 'F':
        sex_female = 1
    else:
        sex_male = 1

    if pclass == '1':
        pclass_1 = 1
    elif pclass == '2':
        pclass_2 = 1
    else:
        pclass_3 = 1

    # Create the DataFrame with the same structure as training data
    input_data = pd.DataFrame([{
        'Age': age,
        'Sex_female': sex_female,
        'Sex_male': sex_male,
        'Pclass_1': pclass_1,
        'Pclass_2': pclass_2,
        'Pclass_3': pclass_3
    }])

    # Load trained model using joblib
    model = joblib.load("titanic.pkl")

    # Use the model to predict probability of survival
    prediction = model.predict_proba(input_data)[0][1]

    return round(prediction * 100, 1)

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Get form data
        age = request.form.get('age')
        sex = request.form.get('sex')
        pclass = request.form.get('pclass')

        # Call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(age, sex, pclass)
            # Pass prediction to template
            return render_template('predict.html', prediction=prediction)

        except ValueError:
            return "Please enter valid values."

    return render_template('titanicPredictor.html')


# Run on Correct Port
if __name__ == '__main__':
    application.debug = True
    application.run(host="localhost", port=5000, debug=True)