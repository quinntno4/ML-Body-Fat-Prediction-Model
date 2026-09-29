AI Body Fat Estimator
Overview
This project is a web-based application that uses machine learning to estimate body fat percentage based on user measurements.  
It also provides calorie recommendations based on activity level and fitness goals.
What it does:
Predicts body fat percentage using a trained ML model
Provides body fat category (e.g. fitness, average, high)
Calculates maintenance calories
Adjusts calories based on user goal (lose, maintain, gain)
Simple web interface
Technologies Used
Python
FastAPI (backend API)
Scikit-learn (machine learning)
Pandas (data handling)
HTML, CSS, JavaScript (frontend)
How to Run the Project
Install dependencies
Open terminal and run:
pip install fastapi uvicorn pandas scikit-learn joblib

Run the backend API
Navigate to the backend folder:
cd backend
Start the API:
uvicorn webapp:app --reload
You should see:
http://127.0.0.1:8000

Open the frontend
Go to:
frontend/index.html
Open the file in your browser (double click or use Live Server).

Use the application
Enter your personal details and body measurements
Click "Calculate Results"
The app will display:
Body fat percentage
Category
Calorie estimates
Notes
The model is trained on publicly available datasets
Results are estimates and should not be used as medical advice
Accuracy may vary depending on the input data
Project Structure
project/
│
├── backend/
│   └── webapp.py
│
├── frontend/
│   ├── index.html
│   └── styles.css
│
├── models/
│   └── bodyfat_model.pkl
│
├── data/
│   └── datasets used for training
│
└── model.py
