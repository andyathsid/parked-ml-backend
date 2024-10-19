import pickle
from flask import Flask, request, jsonify
import numpy as np
from scripts.mdvr_extraction import process_single_file_for_prediction
from sklearn.preprocessing import MinMaxScaler
import os

input_file = 'model_vm_mdvr-kcl_knn.bin'
with open(input_file, 'rb') as file:
    model = pickle.load(file)

print(f'Model loaded from {input_file}')

app = Flask('parkinson-detection')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file:
        file_path = f"./data/raw/{file.filename}"
        file.save(file_path)
    else:
        return jsonify({'error': 'No file provided'}), 400

    processed_data = process_single_file_for_prediction(file_path)
    
    if processed_data is not None:
        X_test = processed_data.values

        scaler = MinMaxScaler()  
        X_test_scaled = scaler.fit_transform(X_test)  

        predictions = model.predict(X_test_scaled)

        probabilities = model.predict_proba(X_test_scaled)

        total_probabilities = probabilities.sum(axis=0)

        final_prediction = np.argmax(total_probabilities)

        result = {
            'Predictions': predictions.tolist(),  
            'Detection': bool(final_prediction)  
        }
    else:
        return jsonify({'error': 'Failed to process the file.'}), 500

    # # Clean up the saved file after processing (optional)
    # if os.path.exists(file_path):
    #     os.remove(file_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
