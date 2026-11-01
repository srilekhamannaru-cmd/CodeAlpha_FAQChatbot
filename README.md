# College Helpdesk FAQ Chatbot

A simple and interactive FAQ chatbot built using Python and Streamlit to help students quickly find answers to common college-related questions.

## About the Project

The College Helpdesk FAQ Chatbot provides information about college facilities and services, including admissions, fees, library timings, examinations, transportation, placements, and more.

It uses Natural Language Processing (NLP) techniques to understand user questions and retrieve the most relevant answers from a predefined FAQ dataset.

## Features

* Interactive chatbot interface.
* Answers 15 predefined college-related FAQs.
* Understands different variations of questions.
* Uses TF-IDF vectorization and cosine similarity.
* Combines word-level and character-level similarity.
* Rejects unrelated questions politely.
* Clear conversation functionality.

## Technologies Used

* Python
* Streamlit
* NLTK
* Scikit-learn
* NumPy
* JSON

## Project Structure

```text
College-Helpdesk-FAQ-Chatbot/
│
├── app.py
├── faqs.json
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd College-Helpdesk-FAQ-Chatbot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. The user enters a question in the chatbot.
2. The question is preprocessed using NLP techniques.
3. TF-IDF converts the question into numerical vectors.
4. Cosine similarity compares the question with stored FAQ questions and variations.
5. The chatbot returns the most relevant answer if the similarity score meets the matching criteria.
6. Unrelated questions receive a polite refusal.

## Future Improvements

* Add more college-related FAQs.
* Support multiple languages.
* Integrate a database for dynamic FAQ management.
* Add voice-based question input.

## Project Status

Completed – All 15 FAQ test questions were successfully tested.
