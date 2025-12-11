# UMBC-DATA606-Capstone

Fake Job Posting Detection Using NLP, LLM Embeddings, and Machine Learning

Prepared for: UMBC Data Science Master’s Degree Capstone
Instructor: Dr. Chaojie (Jay) Wang
Author: Charan Naga Sri Sai Satya Gannapureddy
GitHub Repo: https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone

# 1. Title and Author

## Fake Job Posting Detection Using NLP and Machine Learning

Charan Naga Sri Sai Satya Gannapureddy

GitHub Repository: https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone

LinkedIn: add your link

PowerPoint Presentation: add link

YouTube Presentation: add link

# 2. Background

Online job portals such as LinkedIn, Indeed, and Naukri are increasingly targeted by scammers who post fraudulent job listings. These scams steal personal information or money and cause significant harm to job seekers.

This project uses NLP, Machine Learning, and LLM embeddings to automatically detect fraudulent job postings by analyzing text and metadata.

## Why This Problem Matters

Prevents identity theft and financial loss

Protects job seekers from scams

Helps job portals auto-flag suspicious listings

Improves trust in online recruitment ecosystems

## Research Questions

Can fake job postings be accurately classified using ML and NLP?

Do LLM embeddings outperform TF-IDF?

Which machine learning model performs best?

What textual/metadata features best indicate fraud?

# 3. Data

## 3.1 Data Source

Dataset: Real or Fake Job Posting Prediction

Source: Kaggle

Link: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction

## 3.2 Dataset Description

Size: ~17 MB

Shape: 17,880 rows × 18 columns

Unit of Observation: Each row represents one job posting

## 3.3 Data Structure

Includes:

Text fields: title, description, requirements, company_profile, benefits

Categorical fields: employment_type, salary_range, department

Binary metadata: telecommuting, has_company_logo, has_questions

Target label: fraudulent (1 = fake posting)

## 3.4 Data Dictionary (Key Variables)
| **Column Name**      | **Type**     | **Description**                                     |
| -------------------- | ------------ | --------------------------------------------------- |
| **title**            | text         | Job title                                           |
| **location**         | text         | Geographic location of the job                      |
| **company_profile**  | text         | Description of the hiring company                   |
| **description**      | text         | Main body text describing the job                   |
| **requirements**     | text         | Required skills, qualifications, and experience     |
| **benefits**         | text         | Benefits offered by the employer                    |
| **telecommuting**    | binary (0/1) | Whether the job is remote (1 = yes, 0 = no)         |
| **has_company_logo** | binary (0/1) | Whether the posting includes a company logo         |
| **has_questions**    | binary (0/1) | Whether screening questions were included           |
| **employment_type**  | categorical  | Nature of employment (FT, PT, Contract, Temporary)  |
| **salary_range**     | categorical  | Salary information provided in text format          |
| **fraudulent**       | binary (0/1) | **Target variable**: 1 = Fake job posting, 0 = Real |

### Target Variable
fraudulent
  Binary classification label

  1 = Fake job posting

  0 = Real job posting

### Selected Features

The following features were used to build the machine learning models:

1. Combined Cleaned Text

A unified text field created by merging and preprocessing:

title

company_profile

description

requirements

benefits

This text was later converted into:

TF-IDF vectors (for classical models)

LLM embeddings (MiniLM-L6-v2)

2. Metadata Features

These binary and numeric fields provide structural information:

telecommuting

has_company_logo

has_questions

3. One-Hot Encoded Categorical Variables

Converted into numerical format to support ML models:

employment_type

salary_range

department

industry

required_education

required_experience

function

country (from location)
# 4. Exploratory Data Analysis (EDA)

## 4.1 Data Cleaning

Removed duplicates

Replaced missing text with empty strings

Lowercased text and removed noise (punctuation, symbols, HTML)

## 4.2 Text Preprocessing

Tokenization

Stopword removal

Lemmatization

TF-IDF for classical models

SentenceTransformer embeddings for LLM models

## 4.3 Class Imbalance Handling — SMOTE

Real job postings massively outnumbered fraudulent ones

Applied SMOTE to oversample minority class

Improved model recall and F1-score

## 4.4 Categorical Encoding

One-hot encoded:

department

salary_range

employment_type

country

## 4.5 Key EDA Insights

Fake jobs had vague descriptions

Often lacked company logos

Used repetitive or generic wording

Real postings were longer and more detailed

# 5. Model Training

## 5.1 Models Implemented
Baseline Models (TF-IDF)

Naive Bayes

KNN

Logistic Regression

Random Forest

XGBoost

LLM Embedding Models

Logistic Regression

Random Forest

XGBoost (Best model)

## 5.2 LLM Embedding Approach

Used SentenceTransformer: all-MiniLM-L6-v2

Generates 384-dimensional embeddings

Captures semantic meaning better than TF-IDF

## 5.3 Train–Test Split

80/20 split

Stratified to preserve fraud ratio

## 5.4 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC-AUC

## 5.5 Best Model
⭐ XGBoost + LLM Embeddings

Highest accuracy

Best recall (important for fraud detection)

Most robust across varied text

# 6. Application of the Trained Model

## 6.1 Streamlit Web App

A real-time Streamlit app was created.

Users can:

Paste a job description

Model preprocesses & embeds text

XGBoost predicts fraud probability

App displays classification + confidence

## 6.2 Use Cases

Job portals (auto-flag risky postings)

Universities (protect students from scam listings)

Recruiters (validate external postings)

Job seekers (self-check job authenticity)

# 7. Conclusion
Summary of Results

Machine learning can accurately identify fake job postings

LLM embeddings significantly improved accuracy

XGBoost was the best-performing model

Textual content is the strongest predictor

Limitations

Dataset may not reflect current scam trends

Only English-language data used

No deep transformer fine-tuning due to compute limits

Lessons Learned

Importance of class balancing (SMOTE)

Advantages of LLM embeddings in NLP tasks

Need for thorough preprocessing

Streamlit enables fast, simple deployment

Future Work

Use BERT / RoBERTa for deeper semantic understanding

Add model explainability (SHAP, LIME)

Deploy via REST API

Expand dataset via live scraping

Add multilingual support

# 8. References

Kaggle Fake Job Posting Dataset

Scikit-learn Documentation

SentenceTransformers Documentation

XGBoost Documentation

Imbalanced-Learn (SMOTE)

Research papers on job scam detection
