from flask import Flask, render_template, request, redirect, session, url_for
import joblib

app = Flask(__name__)
app.secret_key = 'ml-project'  # Secret key for session management

# Load the model at the start
@app.route('/') # Home route
def home():
    return render_template('index.html')

@app.route('/step_1', methods=['GET', 'POST'])  # Step 1 route
def step_1():
    if request.method == 'POST':
        session['step_1'] = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
            float(request.form['concave_points_mean']),
            float(request.form['symmetry_mean']),
            float(request.form['fractal_dimension_mean'])
        ]
        return redirect(url_for('step_2'))
    return render_template('mean_values.html')

@app.route('/step_2', methods=['GET', 'POST'])  # Step 2 route
def step_2():
    if request.method == 'POST':
        session['step_2'] = [
            float(request.form['radius_se']),
            float(request.form['texture_se']),
            float(request.form['perimeter_se']),
            float(request.form['area_se']),
            float(request.form['smoothness_se']),
            float(request.form['compactness_se']),
            float(request.form['concavity_se']),
            float(request.form['concave_points_se']),
            float(request.form['symmetry_se']),
            float(request.form['fractal_dimension_se'])
        ]
        return redirect(url_for('step_3'))
    return render_template('se_values.html')

@app.route('/step_3', methods=['GET', 'POST'])  # Step 3 route
def step_3():
    if request.method == 'POST':
        session['step_3'] = [
            float(request.form['radius_worst']),
            float(request.form['texture_worst']),
            float(request.form['perimeter_worst']),
            float(request.form['area_worst']),
            float(request.form['smoothness_worst']),
            float(request.form['compactness_worst']),
            float(request.form['concavity_worst']),
            float(request.form['concave_points_worst']),
            float(request.form['symmetry_worst']),
            float(request.form['fractal_dimension_worst'])
        ]

        all_features = []   # Collect all features from the session
        for i in range(1, 4):   # Loop through steps 1 to 3
            all_features.extend(session[f'step_{i}'])   # Flatten the list

        # Load model and make prediction
        model = joblib.load('optimised_breast_cancer_model.pkl')    # Load the pre-trained model
        prediction = model.predict([all_features])[0]   # Predict the class
        proba = model.predict_proba([all_features])[0].max()    # Get the maximum probability
        label = 'benign' if prediction == 1 else 'malignant'    # Determine the label based on prediction

        return render_template('result.html', label = label, confidence = f"{proba * 100:.2f}%")
    return render_template('worst_values.html')

if __name__ == '__main__':
    app.run(debug=True)