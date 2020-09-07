from flask import Flask,render_template,request,redirect
from nsepy import get_history
from datetime import date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


app= Flask(__name__)

@app.route('/index',methods=['POST','GET'])
def index():
    if request.method=="POST":
        symbol = request.form['symbol']
        #retriving current date
        today = date.today()
        d1 = today.strftime("%Y")
        d2 = today.strftime("%d")
        d3 = today.strftime("%m")
        y=int(d1)
        d=int(d2)
        m=int(d3)
        #retrive stock data

        df= get_history(symbol=symbol , start=date(2020,3,1) ,end=date(y,m,d) )   #year-month-day
        df=df[['Close']]
        #prdicting n days out in future
        forecast_out = 1
        #create another column (to be predicted) prediction
        df['Prediction']=df[['Close']].shift(-forecast_out)

        #print new data set
        #print(df.tail())

        #converting dataframe into numpy array
        X=np.array(df.drop(['Prediction'],1))

        #remove last n rows
        X=X[:-forecast_out]
        #print(X)

        ## create dependent data set i.e convert prediction df into numpy
        #print(type(X))
        Y=np.array(df['Prediction'])
        #remove lasr n row
        Y=Y[:-forecast_out]
        #print(Y)

        #split data into traing n testing
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5)

        x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]

        #print(x_forecast)

        ## train support vector machine regressor
        svr_rbf = SVR(C=1e3,gamma=0.1)
        svr_rbf.fit(X_train,Y_train)

        #print("SVR MODAL PREDICTION")
        ##test n score the above model
        svm_confidence=svr_rbf.score(X_test,Y_test)
        #print("accuracy : " ,svm_confidence)

        #prediction using svr
        #print prediction for next n days
        svm_prediction=svr_rbf.predict(x_forecast)
        #print(f"{x_forecast} next {svm_prediction}")


        #train linnear regression model
        lr=LinearRegression()
        lr.fit(X_train,Y_train)

        #print("linear modal prediction")
        ##test n score the above model
        lr_confidence=lr.score(X_test,Y_test)
        #print("accuracy : " ,lr_confidence)

        #print prediction for next n days
        lr_prediction=lr.predict(x_forecast)
        #print(f"{x_forecast} next {lr_prediction}")

        return render_template('index.html',a1=round(svm_confidence,4),a2=round(lr_confidence,4),p1=svm_prediction,p2=lr_prediction,s=symbol,x=x_forecast)
    else:

        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)