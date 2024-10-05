import joblib
import numpy as np
from flask import Flask, render_template, request

model_pretrained = joblib.load('../loan-or-not-logistic-regression-20240928.pkl')
app = Flask(__name__)
 
@app.route("/")     # root directory
def formPage():
    return render_template('form.html')

 
@app.route("/submit", methods=['GET', 'POST'])
def submit() -> str:
    if request.method == 'POST':
        form_data = request.form

        # credit_history
        credit_history_yes = ''
        credit_history_no = ''
        if int(form_data['credit_history']) == 1:
            credit_history_yes = 'checked'
        else:
            credit_history_no = 'checked'

        # gender
        gender_male = ''
        gender_female = ''
        if int(form_data['gender'])== 1:
            gender_male = 'checked'
        else:
            gender_female = 'checked'

        # married
        married_yes = ''
        married_no = ''
        if int(form_data['married']) == 1:
            married_yes = 'checked'
        else:
            married_no = 'checked'

        # education
        education_graduate = ''
        education_not_graduate = ''
        if int(form_data['education']) == 0:
            education_graduate = 'checked'
        else:
            education_not_graduate = 'checked'

        # dependents
        dependents_0 = ''
        dependents_1 = ''
        dependents_2 = ''
        dependents_3_plus = ''
        if int(form_data['dependents']) == 0:
            dependents_0 = 'selected'
        elif int(form_data['dependents']) == 1:
            dependents_1 = 'selected'
        elif int(form_data['dependents']) == 2:
            dependents_2 = 'selected'
        else:
            dependents_3_plus = 'selected'

        # self_employed
        self_employed_no = ''
        self_employed_yes = ''
        if int(form_data['self_employed']) == 0:
            self_employed_no = 'checked'
        else:
            self_employed_yes = 'checked'

        # property_area
        property_area_rural = ''
        property_area_semiurban = ''
        property_area_urban = ''
        if int(form_data['property_area']) == 0:
            property_area_rural = 'selected'
        elif int(form_data['property_area']) == 1:
            property_area_semiurban = 'selected'
        else:
            property_area_urban = 'selected'

        """
        for testing purpose
        ['credit_history','gender','married','education','dependents','self_employed','property_area','loan_amount_log','total_income_log']
        model_pretrained.predict([[0,1,1,0,3,1,2,np.log(150),np.log(5000)]])
        """
        
        result = model_pretrained.predict([[
            form_data['credit_history'],
            form_data['gender'],
            form_data['married'],
            form_data['education'],
            form_data['dependents'],
            form_data['self_employed'],
            form_data['property_area'],
            np.log(int(form_data['loan_amount'])),
            np.log(int(form_data['total_income']))
        ]])

        result_proba = model_pretrained.predict_proba([[
            form_data['credit_history'],
            form_data['gender'],
            form_data['married'],
            form_data['education'],
            form_data['dependents'],
            form_data['self_employed'],
            form_data['property_area'],
            np.log(int(form_data['loan_amount'])),
            np.log(int(form_data['total_income']))
        ]])

        print(f'Result:{result}')
        print(f'Result_Proba:{result_proba}')

        if result[0] == 1:
            prediction = f'核可(Y) - 系統信心 {result_proba[0][1]:.10f}'    # :.10f represent we wnt to see how many number of digits after DOT
        else:
            prediction = f'拒絕(N) - 系統信心 {result_proba[0][0]:.10f}'

        # reload html
        return render_template(
            'form.html', 
            credit_history_yes=credit_history_yes, 
            credit_history_no=credit_history_no, 
            gender_male=gender_male, 
            gender_female=gender_female, 
            married_yes=married_yes, 
            married_no=married_no, 
            education_graduate=education_graduate, 
            education_not_graduate=education_not_graduate, 
            dependents_0=dependents_0,
            dependents_1=dependents_1,
            dependents_2=dependents_2,
            dependents_3_plus=dependents_3_plus,
            self_employed_no=self_employed_no,
            self_employed_yes=self_employed_yes,
            property_area_rural=property_area_rural,
            property_area_semiurban=property_area_semiurban,
            property_area_urban=property_area_urban,
            loan_amount=form_data['loan_amount'],
            total_income=form_data['total_income'],
            prediction=prediction
        )


if __name__ == "__main__":
    app.run()
