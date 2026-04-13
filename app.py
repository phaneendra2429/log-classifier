from flask import Flask, render_template, request
from src.inference import predict_log

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        log_text = request.form["log"]
        result = predict_log(log_text)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)