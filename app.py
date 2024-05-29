from flask import Flask, render_template, request
import pickle
import numpy as np
from pickle import load

# load the model
model = load(open('model_kmean_preDM.pkl', 'rb'))

# load the scaler
scaler = load(open('scaler_kmean_preDM.pkl', 'rb'))


app = Flask(__name__)  # initializing Flask app

@app.route("/",methods=['GET'])
def hello():
    return (render_template('model_HTML.html'))

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        d1 = request.form['Age']
        d2 = request.form['Height']
        d3 = request.form['Weight']
        d4 = int(d3)/((int(d2)/100)**2)
        d4 = int(d4)
        d5 = request.form['FPG']
        d6 = request.form['HbA1c']
        d7 = request.form['HDL']
        d8 = request.form['ALT']

        
        arr1 = np.array([[d5, d6, d1,d4, d7,d8]]) 

 
        newdata_scaled = scaler.transform(arr1)
        cluster = (model.predict(newdata_scaled))+1

        # df_csv['Cluster'].replace([1,2,3,4,5,6], [3,4,5,1,6,2], inplace  = True) 

        kmeans = ""
        if cluster == 1 :
            kmeans = 'cluster 3'
        elif cluster == 2 :
            kmeans = 'cluster 4'
        elif cluster == 3 :
            kmeans = 'cluster 5'
        elif cluster == 4 :
            kmeans = 'cluster 1'
        elif cluster == 5 :
            kmeans = 'cluster 6'
        elif cluster == 6 :
            kmeans = 'cluster 2'


        char = ""
        if cluster == 4 :
            char = 'Low-risk'
        elif cluster == 6 :
            char = 'Low-risk elderly'
        elif cluster == 1 :
            char = 'Beta cell failure obese'
        elif cluster == 2 :
            char = 'Mild dysglycemia'   
        elif cluster == 3 :
            char = 'High cardiometbolic risk obese' 
        elif cluster == 5 :
            char = 'Severe dysglycemia elderly'     

        print(arr1)
        kmeans = kmeans
        char = char
        return render_template('model_HTML2.html', kmeans=kmeans, char=char )
    else:
        return render_template('model_HTML2.html') 


if __name__ == '__main__':
    app.run()
# app.run(host="0.0.0.0")            # deploy
    # app.run(debug=True)                # run on local system