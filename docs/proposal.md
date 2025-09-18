# Fake Job Posting Prediction: A Machine Learning Approach

**Prepared for**: UMBC Data Science Master’s Degree Capstone  
**Advisor**: Dr. Chaojie (Jay) Wang  

**Author**: Charan Naga Sri Sai Satya Gannapureddy 

- **GitHub Repository**: [https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone](https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone)  
- **LinkedIn Profile**: https://www.linkedin.com/in/satya-gannapureddy/
  

---

## 2. Background  

Online job portals have become a primary channel for job seekers, but they are increasingly exploited by malicious actors who create fraudulent postings. These fake listings aim to steal sensitive information, lure applicants into scams, or extract money under false pretenses. Such scams can have severe consequences for individuals—financial loss, emotional stress, and legal risks—while also damaging the reputation of job platforms. Detecting fraudulent job postings is therefore not only a societal concern but also a critical challenge for industry stakeholders.

This project addresses the problem by applying machine learning and natural language processing (NLP) techniques to a dataset of real and fake job postings. The challenge is unique because it combines structured metadata (e.g., telecommuting status, employment type, company logo) with unstructured text (e.g., descriptions, company profiles). The study aims to build models that not only predict fraudulent postings accurately but also highlight which features—textual or metadata—are most useful in detection.  

- ## Why does it matter?  
Fake job postings can waste time, exploit job seekers, and cause financial or personal harm. Detecting them automatically can help protect users and recruiters.  

- ## Research Questions:  
1. Can job posting text and metadata predict fraudulent postings?  
2. Which features contribute most to distinguishing real job postings from fraudulent ones?
3. To what extent can we reduce false positives in fake job posting detection without sacrificing detection accuracy?  

---

## 3. Data  

- **Source**: [Kaggle – Real or Fake Job Posting Prediction Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
- **Size**: ~17 MB  
- **Shape**: 17,880 rows × 18 columns  
- **Unit of Observation**: Each row = one job posting  

### 📑 Data Dictionary  

| Column Name         | Data Type | Definition                                   | Potential Values |
|----------------------|-----------|-----------------------------------------------|-----------------|
| job_id              | int       | Unique identifier for job                     | 1, 2, 3...      |
| title               | string    | Job title                                     | "Software Eng." |
| location            | string    | Job location                                  | City, State     |
| company_profile      | string    | Company description                           | Free text       |
| description         | string    | Job description text                          | Free text       |
| requirements        | string    | Job requirements                              | Free text       |
| benefits            | string    | Benefits listed                               | Free text       |
| telecommuting       | int       | 1 = remote, 0 = not remote                    | 0, 1            |
| has_company_logo    | int       | Whether logo is present                       | 0, 1            |
| has_questions       | int       | Whether application has screening questions   | 0, 1            |
| employment_type     | string    | Type of job                                   | FT, PT, Contract|
| required_experience | string    | Experience required                           | Entry, Mid, Snr |
| required_education  | string    | Education level required                      | HS, BS, MS, PhD |
| industry            | string    | Industry sector                               | Tech, Finance   |
| function            | string    | Role function                                 | IT, HR, Sales   |
| fraudulent (target) | int       | Label: fake job posting                       | 0 = Real, 1 = Fake |

---

### Target/Label Variable
- **fraudulent** (binary: 0 = Real, 1 = Fake)

### Feature/Predictor Variables
- Text features: `title`, `company_profile`, `description`, `requirements`, `benefits`  
- Metadata: `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`  

---

(👉 You would continue with sections: Methodology, Models, Results, Conclusion, etc.)

