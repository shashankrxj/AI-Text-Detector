from flask import Flask, request, jsonify, render_template
import joblib

# Load the pre-trained unigram models and vectorizer
nb_model_uni = joblib.load('uni model pkl/nb_model.pkl')
lr_model_uni = joblib.load('uni model pkl/lr_model.pkl')
uni_vectorizer = joblib.load('uni model pkl/uni-vectorizer.pkl')

# Load the pre-trained bigram models and vectorizer
rf_model_bi = joblib.load('bi model pkl/rf_model.pkl')
lgb_model_bi = joblib.load('bi model pkl/lgb_model.pkl')
bi_vectorizer = joblib.load('bi model pkl/bi-vectorizer.pkl')

app = Flask(__name__)

# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'unigram')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if model_type == 'unigram':
        # Transform the input text using the unigram vectorizer
        text_vector = uni_vectorizer.transform([text]).toarray()

        # Get predictions from the unigram models
        nb_probs = nb_model_uni.predict_proba(text_vector)[:, 1]
        lr_probs = lr_model_uni.predict_proba(text_vector)[:, 1]

        # Combine predictions (average)
        combined_probs_uni = (nb_probs + lr_probs) / 2
        combined_prediction_uni = (combined_probs_uni > 0.5).astype(int)

        response = {
            'nb_prediction_uni': int(nb_model_uni.predict(text_vector)[0]),
            'lr_prediction_uni': int(lr_model_uni.predict(text_vector)[0]),
            'combined_prediction_uni': int(combined_prediction_uni[0]),
            'nb_probs_uni': float(nb_probs[0]),
            'lr_probs_uni': float(lr_probs[0]),
            'combined_probs_uni': float(combined_probs_uni[0])
        }

    elif model_type == 'bigram':
        # Transform the input text using the bigram vectorizer
        text_vector = bi_vectorizer.transform([text]).toarray()

        # Get predictions from the bigram models
        rf_probs = rf_model_bi.predict_proba(text_vector)[:, 1]
        lgb_probs = lgb_model_bi.predict_proba(text_vector)[:, 1]

        # Combine predictions (average)
        combined_probs_bi = (rf_probs + lgb_probs) / 2
        combined_prediction_bi = (combined_probs_bi > 0.5).astype(int)

        response = {
            'rf_prediction_bi': int(rf_model_bi.predict(text_vector)[0]),
            'lgb_prediction_bi': int(lgb_model_bi.predict(text_vector)[0]),
            'combined_prediction_bi': int(combined_prediction_bi[0]),
            'rf_probs_bi': float(rf_probs[0]),
            'lgb_probs_bi': float(lgb_probs[0]),
            'combined_probs_bi': float(combined_probs_bi[0])
        }

    else:
        return jsonify({"error": "Invalid model type selected"}), 400

    return jsonify(response)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
