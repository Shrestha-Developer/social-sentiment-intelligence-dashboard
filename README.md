🚀 Social Media Sentiment Intelligence Dashboard

An advanced Machine Learning + NLP powered web application built using Streamlit that analyzes social media comments, detects sentiment, categorizes complaints, and generates actionable business insights.

📌 Project Overview

This project transforms raw customer feedback into business intelligence by combining:

🧠 Machine Learning (TF-IDF + Classification Models)
💬 Natural Language Processing (Text Cleaning & Analysis)
📊 Interactive Dashboards (Plotly + Streamlit)
🏢 Business Logic (Priority, Category, Insights)

💡 “Behind every comment is a customer emotion. This dashboard converts public opinion into business intelligence.”

✨ Key Features

🔹 1. Single Comment Analyzer
Predicts sentiment (Positive / Negative / Neutral)
Shows confidence score
Detects complaint category
Suggests business action

🔹 2. Bulk CSV Analyzer
Upload real-world datasets
Auto-detects text column
Handles multiple dataset formats
Generates enriched dataset with:
Sentiment
Confidence
Category
Priority
Recommended Action

🔹 3. Premium Dashboard
📊 Sentiment distribution (Pie Chart)
📈 Trend over time (Line Chart)
📂 Category analysis (Bar Chart)
🧠 Brand Health Score
🎯 Smart filtering (Sentiment, Platform, Category)

🔹 4. Keyword Intelligence
Extracts most frequent words
Identifies customer pain points
Helps in product & marketing decisions

🔹 5. Complaint Monitor
Tracks negative feedback
Highlights high-priority issues
Categorizes complaints

🔹 6. Campaign & Platform Analytics
Compare sentiment across platforms
Analyze campaign performance
Sunburst visualization for insights

🔹 7. ML Model Insights
Compares multiple models:
Logistic Regression
Naive Bayes
Random Forest
Shows:
Accuracy
F1 Score
Confusion Matrix

🔹 8. Downloadable Report
Export fully analyzed dataset
Ready for business reporting


🧠 Machine Learning Pipeline
Raw Text
   ↓
Text Cleaning (Regex, Lowercase, Remove Noise)
   ↓
TF-IDF Vectorization (Unigram + Bigram)
   ↓
Model Training (3 Models)
   ↓
Evaluation (Accuracy + F1 Score)
   ↓
Best Model Selection
   ↓
Prediction + Confidence Score


⚙️ Tech Stack
Category	Tools Used
Frontend UI	Streamlit
Visualization	Plotly
ML Models	Scikit-learn
NLP	TF-IDF, Regex
Data Handling	Pandas, NumPy

📂 Project Structure

📁 project-folder
│
├── app.py                 # Main Streamlit app
├── style.css              # Custom UI styling
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
└── sample_data.csv        # Optional dataset


🚀 Installation & Run
1️⃣ Clone the repository
git clone https://github.com/Shrestha-Developer/social-sentiment-intelligence-dashboard.git
cd social-sentiment-intelligence-dashboard

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app
streamlit run app.py

📥 Dataset Requirements

✅ Required:
At least one column:
text, comment, review, tweet, etc.

➕ Optional:
sentiment
platform
campaign
date

📊 Business Intelligence Logic
Feature	Description
Brand Health	(Positive - Negative) / Total
Priority	Based on sentiment & confidence
Category Detection	Keyword-based classification
Action Engine	Suggests business decisions

🎯 Use Cases
📱 Social Media Monitoring
🛍️ E-commerce Review Analysis
🎯 Marketing Campaign Evaluation
📞 Customer Support Optimization
🏢 Business Intelligence Dashboards

⚠️ Important Note (For Recruiters)
High accuracy on some datasets is due to alignment with training data
Real-world noisy data may reduce performance
This project demonstrates:
End-to-end ML pipeline
Business problem solving
Scalable analytics system

🔮 Future Improvements
🤖 Deep Learning Models (LSTM / BERT)
🌐 Real-time Twitter API integration
📊 Advanced NLP (NER, Topic Modeling)
☁️ Cloud Deployment (AWS / GCP)
📈 User authentication & dashboards

***🙌 Author

Shrestha Mukherjee
Aspiring Data Scientist | ML Engineer | AI Enthusiast ***

⭐ If you like this project

Give it a ⭐ on GitHub and share it!
