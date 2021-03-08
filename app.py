from flask import Flask, request, jsonify, render_template
from converterClass import defaultPredictor


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    output = {}

    if request.method == "POST":

        # convert post request to a dictionary.
        initial_request = {}
        for key, value in request.form.items():
            initial_request[key] = int(value)

        # instantiate defaultPredictor class
        defaultInstance = defaultPredictor(initial_request)
        # converts to pandas frame
        output = defaultInstance.function_converter()
        # call model and make prediction
        prediction = defaultInstance.model_prediction(output)

        return jsonify(prediction)  # jsonify(y_pred_test)

    if request.method == "GET":

        return "This is an API, please request a post to get a result"


if __name__ == "__main__":
    app.run()
