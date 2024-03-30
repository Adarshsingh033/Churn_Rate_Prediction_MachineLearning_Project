from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the XGBoost model
model = pickle.load(open('xgboost_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
    return render_template("churn.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        # Extract form data
        age = int(request.form['age'])
        last_login = int(request.form['last_login'])
        avg_time_spent = float(request.form['avg_time_spent'])
        avg_transaction_value = float(request.form['avg_transaction_value'])
        points_in_wallet = float(request.form['points_in_wallet'])
        date = request.form['date']
        time = request.form['time']
        gender = request.form['gender']
        region_category = request.form['region_category']
        membership_category = request.form['membership_category']
        joined_through_referral = request.form['joined_through_referral']
        preferred_offer_types = request.form['preferred_offer_types']
        medium_of_operation = request.form['medium_of_operation']
        internet_option = request.form['internet_option']
        used_special_discount = request.form['used_special_discount']
        offer_application_preference = request.form['offer_application_preference']
        past_complaint = request.form['past_complaint']
        feedback = request.form['feedback']

        # Perform one-hot encoding for categorical variables
        gender_M = 1 if gender == "M" else 0
        gender_Unknown = 1 if gender == "Unknown" else 0
        region_category_Town = 1 if region_category == "Town" else 0
        region_category_Village = 1 if region_category == "Village" else 0
        membership_category_Gold = 1 if membership_category == "Gold Membership" else 0
        membership_category_No = 1 if membership_category == "No Membership" else 0
        membership_category_Platinum = 1 if membership_category == "Platinum Membership" else 0
        membership_category_Silver = 1 if membership_category == "Silver Membership" else 0
        membership_category_Premium = 1 if membership_category == "Premium Membership" else 0
        joined_through_referral_No = 1 if joined_through_referral == "No" else 0
        joined_through_referral_Yes = 1 if joined_through_referral == "Yes" else 0
        preferred_offer_types_Gift_VouchersCoupons = 1 if preferred_offer_types == "Gift Vouchers/Coupons" else 0
        preferred_offer_types_Without_Offers = 1 if preferred_offer_types == "Without Offers" else 0
        medium_of_operation_Both = 1 if medium_of_operation == "Both" else 0
        medium_of_operation_Desktop = 1 if medium_of_operation == "Desktop" else 0
        medium_of_operation_Smartphone = 1 if medium_of_operation == "Smartphone" else 0
        internet_option_Mobile_Data = 1 if internet_option == "Mobile_Data" else 0
        internet_option_Wi_Fi = 1 if internet_option == "Wi-Fi" else 0
        used_special_discount_Yes = 1 if used_special_discount == "Yes" else 0
        offer_application_preference_Yes = 1 if offer_application_preference == "Yes" else 0
        past_complaint_Yes = 1 if past_complaint == "Yes" else 0

        # Perform one-hot encoding for feedback
        feedback_labels = [
            "Poor Customer Service",
            "Poor Product Quality",
            "Poor Website",
            "Products always in Stock",
            "Quality Customer Care",
            "Reasonable Price",
            "Too many ads",
            "User Friendly Website"
        ]
        feedback_values = [0] * len(feedback_labels)
        if feedback in feedback_labels:
            feedback_values[feedback_labels.index(feedback)] = 1

        # Extract date and time components
        joining_day, joining_month, joining_year = map(int, date.split("-"))
        last_visit_time_hour, last_visit_time_minutes, last_visit_time_seconds = map(int, time.split(":"))

        # Create a DataFrame for prediction
        data = {
            "age": [age],
            "days_since_last_login": [last_login],
            "avg_time_spent": [avg_time_spent],
            "avg_transaction_value": [avg_transaction_value],
            "points_in_wallet": [points_in_wallet],
            "joining_day": [joining_day],
            "joining_month": [joining_month],
            "joining_year": [joining_year],
            "last_visit_time_hour": [last_visit_time_hour],
            "last_visit_time_minutes": [last_visit_time_minutes],
            "last_visit_time_seconds": [last_visit_time_seconds],
            "gender_M": [gender_M],
            "gender_Unknown": [gender_Unknown],
            "region_category_Town": [region_category_Town],
            "region_category_Village": [region_category_Village],
            "membership_category_Gold Membership": [membership_category_Gold],
            "membership_category_No Membership": [membership_category_No],
            "membership_category_Platinum Membership": [membership_category_Platinum],
            "membership_category_Premium Membership": [membership_category_Premium],
            "membership_category_Silver Membership": [membership_category_Silver],
            "joined_through_referral_No": [joined_through_referral_No],
            "joined_through_referral_Yes": [joined_through_referral_Yes],
            "preferred_offer_types_Gift Vouchers/Coupons": [preferred_offer_types_Gift_VouchersCoupons],
            "preferred_offer_types_Without Offers": [preferred_offer_types_Without_Offers],
            "medium_of_operation_Both": [medium_of_operation_Both],
            "medium_of_operation_Desktop": [medium_of_operation_Desktop],
            "medium_of_operation_Smartphone": [medium_of_operation_Smartphone],
            "internet_option_Mobile_Data": [internet_option_Mobile_Data],
            "internet_option_Wi-Fi": [internet_option_Wi_Fi],
            "used_special_discount_Yes": [used_special_discount_Yes],
            "offer_application_preference_Yes": [offer_application_preference_Yes],
            "past_complaint_Yes": [past_complaint_Yes],
        }
        data.update({"feedback_" + label: [value] for label, value in zip(feedback_labels, feedback_values)})
        df = pd.DataFrame(data)

        # Perform prediction using the loaded XGBoost model
        prediction = model.predict(df)

        # Render the prediction result in the template
        return render_template("prediction.html", prediction_text="Churn Score is {}".format(prediction))

    else:
        return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)
