# Emotion-classifier-with-transformers

This project demonstrates the use of transformer models to classify emotions in text data. Leveraging the distilbert-base-uncased model from Hugging Face's transformers library, this project aims to accurately predict emotions such as sadness, joy, love, anger, fear, and surprise from given text inputs. The project includes steps for data preprocessing, model training, evaluation, and visualization of results.

# Project Workflow

# 1- Setup and Initialization:

Load the pre-trained DistilBERT model and tokenizer.
Ensure the model and data tensors are transferred to the appropriate device (GPU if available).

# 2- Data Preprocessing:

Tokenize the input text data with padding and truncation.
Encode the tokenized text data into the appropriate format for model input.

# 3- Feature Extraction:

Extract hidden states from the model's last hidden layer.
Use the hidden state of the [CLS] token as feature vectors representing each text input.

# 4- Dimensionality Reduction and Visualization:

Apply UMAP for dimensionality reduction to visualize the high-dimensional feature vectors in 2D space.
Generate scatter plots to visualize the distribution of different emotion classes.

![image](https://github.com/elkomy13/Emotion-classifier-with-transformers/assets/97259226/d2118318-9aee-47cf-8628-5f3ad38e385e)


# 5- Training and Evaluation:

Train a logistic regression model on the extracted features.
Evaluate the logistic regression model using accuracy and compare it with a baseline dummy classifier.
Fine-tune the DistilBERT model for sequence classification with the emotion labels.
Use the Trainer class from transformers for efficient training and evaluation.
Compute and log evaluation metrics such as accuracy and F1-score.

# 6- Prediction and Loss Computation:

Perform predictions on the validation set using the fine-tuned model.
Compute cross-entropy loss for each validation example to analyze the model performance.

# Largest Loss :
![image](https://github.com/elkomy13/Emotion-classifier-with-transformers/assets/97259226/594bb897-b3b3-49ce-aa5a-35ddf00771ea)

# Smallest Loss :
![image](https://github.com/elkomy13/Emotion-classifier-with-transformers/assets/97259226/3503d9f3-ef32-4007-9ae7-96eacee4af3d)

