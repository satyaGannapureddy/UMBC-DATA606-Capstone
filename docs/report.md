# UMBC-DATA606-Capstone

Fake Job Posting Detection Using NLP, LLM Embeddings, and Machine Learning

Prepared for: UMBC Data Science Master’s Degree Capstone

Instructor: Dr. Chaojie (Jay) Wang

Author: Charan Naga Sri Sai Satya Gannapureddy


# Fake Job Posting Detection Using NLP and Machine Learning

Charan Naga Sri Sai Satya Gannapureddy

GitHub Repository: https://github.com/satyaGannapureddy/UMBC-DATA606-Capstone

LinkedIn: https://www.linkedin.com/in/satya-gannapureddy/

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
  + Binary classification label

   + 1 = Fake job posting

   + 0 = Real job posting

### Selected Features

The following features were used to build the machine learning models:

#### 1. Combined Cleaned Text

A unified text field created by merging and preprocessing:

+ title

+ company_profile

+ description

+ requirements

+ benefits

This text was later converted into:

+ TF-IDF vectors (for classical models)

+ LLM embeddings (MiniLM-L6-v2)

#### 2. Metadata Features

These binary and numeric fields provide structural information:

+ telecommuting

+ has_company_logo

+ has_questions

#### 3. One-Hot Encoded Categorical Variables

Converted into numerical format to support ML models:

+ employment_type

+ salary_range

+ department

+ industry

+ required_education

+ required_experience

+ function

+ country (from location)

# 4. Exploratory Data Analysis (EDA)

The exploratory data analysis was performed in Jupyter Notebook to understand the structure, quality, and behavior of the Fake Job Posting dataset before applying machine learning. The dataset contains 17,880 job postings with 18 columns, including text fields, categorical variables, and binary indicators. For analysis, we focused on the target variable fraudulent and a selected set of features such as description, title, company_profile, requirements, benefits, telecommuting, has_company_logo, has_questions, employment_type, salary_range, required_experience, required_education, industry, and function. These variables were retained because they provide meaningful signals for fraud detection, while unrelated identifiers were dropped.

Initial summary statistics revealed that real job postings tend to contain longer, richer text descriptions, while fake postings often provide vague or short descriptions. Binary metadata showed useful patterns: for example, fake postings rarely include company logos, and they also tend not to ask screening questions, suggesting weaker legitimacy signals. Categorical variables such as job function, industry, and employment type exhibited substantial imbalance, and many missing entries were consolidated under “Unknown” to maintain data integrity.

Data cleaning involved handling missing values—text fields were filled with empty strings, and categorical fields were assigned “Unknown.” Duplicate rows were checked and removed when present. No merging, pivoting, or external datasets were required because the dataset is self-contained for the prediction task. The location field was split to extract the country component, enabling basic geographic distribution analysis. After preprocessing, the dataset was made “tidy”: each row represents a single job posting, and each column represents one unique attribute of that posting.

Text preprocessing included lowercasing, removing punctuation, removing stopwords, tokenizing, and lemmatizing. A combined cleaned text feature was created by merging multiple text fields (title, description, requirements, etc.), which was later vectorized using TF-IDF and LLM embeddings.

Below are the visualizations included as part of EDA:

#### Class Imbalance


<img src="images/class imblance.png">


Fake postings make up only about 4.8% of all listings, indicating a strong class imbalance that requires oversampling (SMOTE).

#### Binary Metadata Distributions


<img src="images/meta data distributions.png">


These plots show that fake postings rarely include company logos and often skip screening questions, whereas legitimate postings usually provide them.

#### Job Function Distribution


<img src="images/Job Function Distribution.png">


The majority of postings belong to “Unknown” or broad categories like IT and Sales; many categories appear infrequently.

#### Correlation Heatmap


<img src="images/heatmap.png">


The heatmap shows that has_company_logo has a noticeable negative correlation with fraud, while has_questions has a slight positive association with real postings.

#### Country Distribution


<img src="images/Country Distribution.png">


Most postings originate from the US, followed by GB and GR; other countries appear sparsely.

#### Word Count in Job Descriptions


<img src="images/Country Distribution.png">


Real postings generally contain longer descriptions compared to fake postings, which helps models differentiate them.

#### Character Count in Job Descriptions


<img src="images/Character Count in Job Descriptions.png">


Real postings show a wide spread of character counts, while fake postings cluster at low values.

#### Word Cloud of Common Terms


<img src="images/word count.png">


The word cloud reveals recurring words such as communication, experience, customer service, and full time, suggesting typical job-related terminology.

# 5. Model Training
To build a predictive system for identifying fake job postings, several machine learning models were trained and evaluated using both traditional NLP representations and modern embedding-based techniques. The initial stage involved preparing the dataset by performing text preprocessing (normalization, stopword removal, lemmatization) and converting text into numerical form using TF-IDF vectorization. Classical models such as Naive Bayes, K-Nearest Neighbors (KNN), Logistic Regression, Random Forest, and XGBoost were then trained on this representation. The dataset was split using an 80/20 stratified train–test split to maintain the proportion of fraudulent postings in both sets, ensuring fair and consistent evaluation.

Python libraries including scikit-learn, NLTK, spaCy, XGBoost, pandas, NumPy, and Matplotlib were used for preprocessing, modeling, and visualization. All model development was carried out in a Jupyter Notebook environment running locally. After observing limitations in TF-IDF models—particularly in detecting the minority (fraudulent) class—the pipeline was extended using SentenceTransformer embeddings (all-MiniLM-L6-v2). This embedding model generates 384-dimensional contextual vectors that capture richer semantic meaning compared to TF-IDF.

##### Comparison of Model Accuracies (TF-IDF Models)


<img src="images/model accuarices.png">


The bar chart shows how five traditional machine learning models performed when trained using TF-IDF features. There is a noticeable difference between the simpler models and the more advanced ones. Naive Bayes clearly struggles, reaching only about 39% accuracy, which suggests that it cannot capture the complexity and variability of job-posting text.

The other models — KNN, Logistic Regression, Random Forest, and XGBoost — perform much better, all landing in the 85% to 89% accuracy range. Among them, Random Forest (0.895) and KNN (0.893) come out on top, with Logistic Regression and XGBoost close behind. These results show that traditional ML models can learn meaningful patterns from the text, but they still face challenges.

Even though the accuracies look relatively high, the dataset is highly imbalanced, with real postings vastly outnumbering fake ones. Because of this imbalance, accuracy alone doesn’t tell the whole story. Many of these models still miss a large portion of fraudulent postings, which becomes clearer when we look at recall and F1-scores later.

Overall, this comparison highlights that while TF-IDF models perform reasonably well on surface-level accuracy, more advanced text representations — such as LLM-based embeddings — are necessary to truly improve the detection of fake job postings.

##### Confusion Matrix Components Comparison (TF-IDF Models)


<img src="images/Confusion matrix.png">

This chart provides a deeper look at how each TF-IDF model performs by breaking down the confusion matrix components: true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). This view is especially important for fraud detection because accuracy alone can hide how poorly a model identifies the minority class.

A clear pattern emerges across the models. Naive Bayes performs the worst, producing a very high number of false positives and very few true positives. This means it incorrectly flags many real job postings as fake while still missing most of the actual fraudulent ones.

The other models—KNN, Logistic Regression, Random Forest, and XGBoost—show much stronger performance by correctly identifying a large number of true negatives (TN), which is expected given the dataset imbalance. However, the real test is how well they detect fraudulent postings. While these models do capture more true positives (TP) than Naive Bayes, they still produce a significant number of false negatives (FN), meaning many fake postings go undetected.

Among the TF-IDF models, XGBoost and Logistic Regression strike the best balance, showing relatively higher true positive counts and fewer false negatives compared to the others. Still, the chart makes it clear that all traditional models struggle with identifying fraud effectively, reinforcing the need for improved text representations—such as LLM embeddings—to boost fraud detection performance.

##### Accuracy by Model (LLM Embeddings)


<img src="images/accuracy by model.png">

This chart compares the accuracy of three machine learning models—XGBoost, Random Forest, and Logistic Regression—after replacing traditional TF-IDF features with LLM-generated embeddings (MiniLM-L6-v2). The improvement is immediately visible. All three models achieve accuracy scores above 96%, with LLM + XGBoost performing the best at 97.5%.

Compared to the earlier TF-IDF models, which topped out around 89%, this represents a major performance boost. The reason is that LLM embeddings capture deeper semantic meaning from the job descriptions, allowing the models to better differentiate between subtle patterns in real vs. fraudulent postings.

While accuracy alone doesn't tell the full story—especially in an imbalanced dataset—this plot demonstrates how switching to LLM-based features significantly enhances the model's overall predictive capability. The consistently high accuracy across all three models also shows that embedding-based representations provide a strong universal foundation, regardless of the classifier used.

##### Macro-F1 by Model (LLM Embeddings)


<img src="images/marco F1.png"> 

The Macro-F1 chart provides a more meaningful comparison of model performance on this imbalanced dataset by averaging the F1-scores of both classes equally. Unlike accuracy, which can appear high even when the model fails on the minority class, the Macro-F1 score highlights how well each model balances precision and recall for both real and fraudulent job postings.

From the results, LLM + XGBoost stands out with the highest Macro-F1 score of 0.829, showing that it is the most effective at identifying both legitimate and fraudulent postings without being overly biased toward the majority class. Random Forest (0.752) and Logistic Regression (0.741) also perform well, but both fall behind XGBoost in terms of capturing fraud cases reliably.

Overall, this comparison confirms that LLM embeddings significantly improve the models’ ability to detect fraudulent postings, and XGBoost benefits the most from these richer text representations. This makes LLM + XGBoost the strongest model not only in accuracy but also in balanced performance across both classes.


# 6. Application of the Trained Model

## 6.1 Streamlit Web App
To make the fraud detection system practical and easy to use, a real-time web application was developed using Streamlit. The application allows users to interact directly with the trained LLM + XGBoost model and receive instant predictions on whether a job posting is likely to be fraudulent.

The app supports single job predictions through a user-friendly form interface. Users can enter key job-related information such as location, department, salary range, employment type, and the full job description text. Additional metadata fields like telecommuting status, presence of a company logo, and screening questions can also be selected. Once the user clicks the Predict button, the input text is preprocessed and converted into LLM embeddings, combined with encoded metadata, and passed to the trained XGBoost classifier.

The model outputs both a binary classification (Fraudulent / Not Fraudulent) and a fraud probability score, giving users a clear and interpretable result rather than just a yes-or-no answer.

<img src="docs/marco F1.png"> 

After prediction, the application displays the result prominently, highlighting whether the job posting is considered fraudulent. A confidence score (fraud probability) is also shown, helping users understand how strongly the model believes in its prediction. The model configuration used for prediction—LLM embeddings (MiniLM-L6-v2) combined with XGBoost—is clearly mentioned for transparency.

<img src="docs/marco F1.png"> 

## 6.2 Use Cases

This application demonstrates how machine learning models can be translated into real-world tools with meaningful impact. Potential use cases include:

Job portals: Automatically flag suspicious job postings before they reach users

Universities: Protect students from scam listings targeting early-career job seekers

Recruiters: Verify the legitimacy of external job postings shared on platforms

Job seekers: Independently check the authenticity of job postings before applying

Overall, the Streamlit application bridges the gap between model development and real-world usability, showcasing how advanced NLP and machine learning techniques can be deployed as an accessible and practical solution for fraud detection.

# 7. Conclusion
### 7.1 Project Summary & Impact

This project focused on detecting fraudulent job postings using machine learning and natural language processing techniques. By analyzing both textual job descriptions and structured metadata, the study demonstrated that machine learning models can effectively distinguish between real and fake job postings. While traditional TF-IDF–based models achieved reasonable accuracy, they struggled with identifying fraudulent postings due to strong class imbalance.

The introduction of LLM-based embeddings significantly improved model performance by capturing deeper semantic meaning in job descriptions. Among all approaches tested, XGBoost combined with LLM embeddings produced the strongest and most balanced results, achieving high accuracy and Macro-F1 scores. The final Streamlit application further translated this model into a practical tool that allows users to assess job postings in real time, helping protect job seekers, educational institutions, and recruiting platforms from online scams.

### 7.2 Limitations

Despite its strong performance, this project has several limitations. The dataset used may not fully reflect modern or evolving scam strategies, as fraudulent job postings continuously adapt to bypass detection systems. Additionally, the analysis was restricted to English-language postings, limiting its applicability to global job markets. Computational constraints also prevented fine-tuning large transformer models such as BERT or RoBERTa, which may have yielded further improvements in performance.

### 7.3 Lessons Learned

This project highlighted several important lessons in applied machine learning. Handling class imbalance using techniques such as SMOTE was critical for improving fraud detection performance. The results clearly demonstrated the advantages of LLM embeddings over traditional TF-IDF representations for text-heavy classification tasks. Careful text preprocessing and feature engineering played a major role in model reliability, and deploying the final model using Streamlit showed how quickly machine learning research can be converted into a usable, real-world application.

### 7.4 Future Directions

Future work could focus on incorporating transformer-based models such as BERT or RoBERTa for deeper semantic understanding of job descriptions. Adding model explainability tools like SHAP or LIME would improve transparency and trust in predictions. Deploying the system as a REST API would enable easier integration with job portals and external applications. Expanding the dataset through live job scraping and supporting multilingual job postings would further enhance the system’s robustness and real-world impact.


## 8. References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [SentenceTransformers Documentation](https://www.sbert.net/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Kaggle – Real or Fake Job Posting Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)
- [spaCy Documentation](https://spacy.io/usage)




