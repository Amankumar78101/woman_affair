from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            rate_marriage=float(request.form.get('Rate Marriage')),
            age = float(request.form.get('AGE')),
            yrs_married = float(request.form.get('Year Married')),
            children = float(request.form.get('children')),
            religious = float(request.form.get('religious')),
            educ = float(request.form.get('educ')),
            occupation = float(request.form.get('occupation')),
            occupation_husb= float(request.form.get('occupation_husb'))
            )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

