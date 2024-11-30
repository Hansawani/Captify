# CaptifyCaptify: Social Media Caption Generator
Captify is an AI-powered platform designed to generate engaging, creative, and contextually relevant captions for social media images. It leverages state-of-the-art AI models and a robust preprocessing pipeline to provide users with captions enriched with hashtags, emojis, and trendy language, tailored specifically for platforms like Instagram.

Features
Mood and Aesthetic Analysis: Uses CLIP to extract emotional and stylistic labels from images.
Descriptive Captioning: Employs BLIP to generate context-aware image descriptions.
Caption Synthesis: Integrates CLIP and BLIP outputs into a fine-tuned Qwen2-7B model to create captions optimized for social media.
User-Friendly Interface: Developed with Streamlit for a seamless, interactive experience.
Tech Stack
Programming Language:
Python 3.10
Libraries and Frameworks:
Data Handling:
pandas, os, zipfile
Deep Learning Models:
transformers, datasets, torch, torchvision, accelerate
Pretrained Models:
CLIP (openai/clip-vit-base-patch32)
BLIP (Image Captioning)
Qwen2-7B (Fine-tuned for social media-style captions)
Image Processing:
PIL
Model Fine-Tuning Framework:
MS-SWiFT (Includes PEFT techniques like LoRA)
Frontend Development:
Streamlit
Dataset
The dataset used in this project was downloaded from Kaggle: Instagram Images with Captions (https://www.kaggle.com/datasets/prithvijaunjale/instagram-images-with-captions)

Dataset Setup:
Download the dataset from the Kaggle link above.

Store the zipped dataset (instagram_data.zip) in Google Drive.

Extract and preprocess captions:
![image](https://github.com/user-attachments/assets/1b4d6a4e-5183-4ea2-9e4e-a5c4a7d5e614)


Load the captions file:
![image](https://github.com/user-attachments/assets/14246967-dde3-4a8c-abe8-c0861f63e8e8)

Setup and Installation

Prerequisites:
Google Drive account to store datasets and processed files.
Access to a GPU-enabled environment like Google Colab.

Steps to Install:
Clone or download the repository containing the notebook.
Install the required libraries:
Mount Google Drive to access the datasets:

Workflow
Image Preprocessing: Images are resized, normalized, and prepared for analysis.
Feature Extraction:
CLIP identifies mood and aesthetic labels through cosine similarity scoring.
BLIP generates detailed descriptions for image content.

Fine-Tuning:
The outputs from CLIP and BLIP are merged into prompt-response pairs.
The Qwen2-7B model is fine-tuned using MS-SWiFT with LoRA techniques for efficient training.

Caption Generation: The final captions are generated dynamically based on multimodal inputs.
User Interaction: Captions can be customized and rated through a Streamlit interface.

Running the Notebook
Launch the notebook in Google Colab.
Mount Google Drive and ensure the dataset path is correct.
Execute the cells sequentially to preprocess data, fine-tune the model, and generate captions.

Acknowledgments
We express our gratitude to:

Kaggle for hosting the dataset: Instagram Images with Captions.
Hugging Face for pre-trained models.
ModelScope for the MS-SWiFT framework.
Open-source contributors for libraries like pandas, torch, and Streamlit.

Future Work
Expand datasets to include diverse platforms like Twitter and Pinterest.
Add multilingual caption support.
Integrate advanced user feedback loops for adaptive learning.
Enhance hashtag optimization and trend tracking.
