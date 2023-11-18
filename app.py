from flask import Flask, request, jsonify
from flask_cors import CORS  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)  
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year_to_predict = data['year']
    
    df = pd.read_csv('regression_data.csv')
    df = df.iloc[1:, 0]
    train = df.iloc[:-3]
    test = df.iloc[-3:]
    yframe = pd.DataFrame({'Year': range(2003, 2012)})
    train_data = pd.Series(train.values)
    y_data = pd.Series(yframe.values.flatten().tolist(), name='Year')

    train_data = train_data.str.replace(',', '').astype(float)
    df = pd.DataFrame({'X': train_data, 'Year': y_data})
    X_train, X_test, y_train, y_test = train_test_split(df[['Year']], df['X'], test_size=0.2, random_state=42)
    degree = 8
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    ridge_model = LinearRegression() 
    ridge_model.fit(X_train_poly, y_train)
    
    X_to_predict_poly = poly_features.transform(np.array([[year_to_predict]]))
    predicted_X_poly = ridge_model.predict(X_to_predict_poly)
    
    result = {
        'year': year_to_predict,
        'predicted_X': round(predicted_X_poly[0])
    }
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
