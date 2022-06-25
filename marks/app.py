from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/percent', methods = ["POST","GET"])
def percent():
    if request.method == "POST":
        Study = request.form("study")
        Age = request.form("age")
        Internet = request.form("internet")
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    dataset = pd.read_csv('data.csv')
    dataset.columns[dataset.isna().any()]
    dataset.hours = dataset.hours.fillna(dataset.hours.mean())
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    model = LinearRegression()
    model.fit(X,Y)
    inputs=[[Study,Age,Internet]]
    output = float(model.predict(inputs))
    marks = str(round(output, 2))
    per = ' %'
    return render_template("index.html", percent=marks, symbol=per)

if __name__ == "__main__":
    app.run()