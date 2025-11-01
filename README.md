🛡️ Phishing URL Detection Using Machine Learning

Created by Abhishek & Priyanka

This project uses Machine Learning to detect phishing URLs based on their characteristics and patterns. It analyzes features like domain, prefix/suffix, HTTPS usage, and traffic to classify whether a given URL is safe or malicious.

📋 Table of Contents

Introduction

Installation

Directory Tree

Technologies Used

Results

Conclusion

🧠 Introduction

The Internet is a major part of our daily life — but it also provides attackers with ways to perform phishing attacks, where fake websites are created to steal sensitive data such as usernames, passwords, or banking information.

To fight this, we use Machine Learning algorithms to automatically detect phishing URLs by analyzing their structural and behavioral features.

👉 This project applies various ML models and compares their performance to identify the most accurate one.

⚙️ Installation

This project is built using Python 3.8+.
To install all dependencies, open your terminal inside the project folder and run:

pip install -r requirements.txt


If you don’t have Python installed, you can download it from python.org
.

📁 Directory Tree
├── pickle
│   ├── model.pkl
├── static
│   ├── styles.css
├── templates
│   ├── index.html
├── Phishing URL Detection.ipynb
├── Procfile
├── README.md
├── app.py
├── feature.py
├── phishing.csv
├── requirements.txt

🧰 Technologies Used
Category	Tools & Libraries
Language	Python
ML Libraries	Scikit-learn, XGBoost, CatBoost, RandomForest
Data Handling	Pandas, NumPy
Visualization	Matplotlib
Web Framework	Flask
Model Storage	Pickle
📊 Results

Accuracy comparison of different ML models used for phishing URL detection:

#	Model	Accuracy	F1-Score	Recall	Precision
1	Gradient Boosting Classifier	0.974	0.977	0.994	0.986
2	CatBoost Classifier	0.972	0.975	0.994	0.989
3	XGBoost Classifier	0.969	0.973	0.993	0.984
4	Multi-layer Perceptron	0.969	0.973	0.995	0.981
5	Random Forest	0.967	0.971	0.993	0.990
6	SVM (Support Vector Machine)	0.964	0.968	0.980	0.965
7	Decision Tree	0.960	0.964	0.991	0.993
8	K-Nearest Neighbors	0.956	0.961	0.991	0.989
9	Logistic Regression	0.934	0.941	0.943	0.927
10	Naive Bayes	0.605	0.454	0.292	0.997
🔍 Feature Importance

🧾 Conclusion

This project explores multiple ML models for phishing URL detection, including boosting and ensemble techniques.

It provides insights into how feature selection and model tuning affect accuracy.

Features like HTTPS, AnchorURL, and WebsiteTraffic play key roles in classification.

The Gradient Boosting Classifier achieved the highest accuracy of 97.4%, making it the best performer for detecting phishing websites.

👩‍💻 Authors

Abhishek & Priyanka
📧 abhisheknishad374@gmail.com
📧 jhap3187@gmail.com
