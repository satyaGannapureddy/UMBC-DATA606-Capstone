# Employment Fraud Detection System

**DATA 606**: Capstone Project  
**Faculty**: Dr. Chaojie (Jay) Wang  

**Author**: Charan Naga Sri Sai Satya Gannapureddy 

- **GitHub Repository**: [https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone](https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone)  
- **LinkedIn Profile**: https://www.linkedin.com/in/satya-gannapureddy/
  

---

## 2. Background  

Online job portals have become a primary channel for job seekers, but they are increasingly exploited by malicious actors who create fraudulent postings. These fake listings aim to steal sensitive information, lure applicants into scams, or extract money under false pretenses. Such scams can have severe consequences for individuals financial loss, emotional stress, and legal risks while also damaging the reputation of job platforms. Detecting fraudulent job postings is therefore not only a societal concern but also a critical challenge for industry stakeholders.

This project addresses the problem by applying machine learning and natural language processing (NLP) techniques to a dataset of real and fake job postings.The study aims to build models that not only predict fraudulent postings accurately but also highlight which features textual or metadata are most useful in detection.  

## Why does it matter?  
Fake job postings can have serious consequences. Many scams are designed to steal personal information, leading to identity theft or financial loss, and they can leave job seekers emotionally drained. On the other side, genuine recruiters also suffer when their platforms are misused by fraudsters, as it damages their reputation. For job portals, the constant presence of fake listings erodes user trust and forces them to spend more on monitoring and security. That’s why building automated systems to detect these scams is so important it protects job seekers, helps recruiters, and keeps online hiring platforms safe and reliable. 

## Research Questions:  
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
| department          | string    | Department within the company                 | "Engineering", "Sales", ... |
| salary_range        | string    | Salary range offered                           | "50k-70k", "100000+", "Negotiable" |
| company_profile      | string    | Company description                          |text             |
| description         | string    | Job description text                          |text             |
| requirements        | string    | Job requirements                              |text             |
| benefits            | string    | Benefits listed                               |text             |
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

All columns except `job_id` (identifier) and `fraudulent` (target) are used as predictors. Key features include:  

- **Text fields**: title, company_profile, description, requirements, benefits  
- **Categorical fields**: department, salary_range, employment_type, required_experience, required_education, industry, function, location  
- **Binary fields**: telecommuting, has_company_logo, has_questions  




