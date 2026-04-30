# Chatbot Intent Classification with Hugging Face Transformers

This repository provides a complete NLP pipeline for chatbot intent classification using a fine-tuned BERT model.

## Project Overview

This project aims to classify user utterances into predefined intents using state-of-the-art NLP techniques. The pipeline includes:

*   **Dataset:** Utilizing the `snips_built_in_intents` dataset from Hugging Face for multi-class text classification.
*   **Preprocessing:** Tokenization and formatting of text data for transformer models.
*   **Baseline Model:** Implementation of a TF-IDF + Multinomial Naive Bayes model for comparison.
*   **Transformer Fine-tuning:** Fine-tuning a `bert-base-uncased` model for superior performance.
*   **Evaluation:** Comprehensive assessment using accuracy, precision, recall, and F1-score.
*   **Inference & Deployment:** A simple interactive interface built with Gradio for real-time intent prediction.

## How to Run

To get started with this project in Google Colab:

1.  **Install Dependencies:**
    ```bash
    !pip install -r requirements.txt
    ```
    The `requirements.txt` includes `transformers`, `datasets`, `torch`, `gradio`, `scikit-learn`, `evaluate`, `seaborn`, and `matplotlib`.

2.  **Open the Notebook:**
    Open the `Chatbot Intent Classification with Hugging Face Transformers.ipynb` notebook in Google Colab.

3.  **Execute All Cells:**
    Run all the cells sequentially. This will handle dataset loading, model training, evaluation, and launch a Gradio application for interactive testing.

## Key Technologies Used

*   **Hugging Face Transformers:** For pre-trained models and fine-tuning (`BERT`).
*   **Hugging Face Datasets:** For efficient dataset loading and management.
*   **PyTorch:** The deep learning framework powering the transformer model.
*   **scikit-learn:** For baseline modeling and evaluation metrics.
*   **Gradio:** For creating an interactive web demo.
*   **Python:** The primary programming language.

