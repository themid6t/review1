📜 NativeGPT Project Presentation Script
🧑‍💼 Team Members:
1️⃣ Tabarakul Islam Hazarika (Lead Presenter)
2️⃣ Uday Sankor Mohan
3️⃣ Trishita Das
4️⃣ Changrung Modhur
5️⃣ Rishikesh Sonowal

⏳ Total Time: ~10 minutes
## 🎙️ 1. Introduction – Tabarakul (Lead Presenter) (1.5 min)
*"Good [morning/afternoon/evening] everyone. We are excited to present our project—NativeGPT, an AI assistant designed to understand and generate Assamese text fluently.

The world of AI is expanding rapidly, but regional languages, especially Assamese, are underrepresented. Our goal is to bridge this gap by fine-tuning an open-source language model specifically for Assamese chat applications.

Over the past few months, we have worked extensively on dataset preparation, model selection, training pipeline setup, and building a real-time chat interface. Today, we will take you through our project’s journey, challenges, and future scope. Let's begin!"*

## 🎙️ 2. Problem Statement & Motivation – Uday (2 min)
*"India is home to over 1,600 languages, but most AI models today primarily support English, Hindi, and a few other widely spoken languages.

However, Assamese, which has over 23 million speakers, lacks a robust AI model for chat-based applications. Current AI models either fail to generate meaningful Assamese text or provide highly inaccurate responses. This limits the accessibility of AI-powered applications for Assamese users.

NativeGPT aims to solve this by fine-tuning an AI model that can understand Assamese syntax, grammar, and nuances. This will have significant applications in customer service, education, and digital communication, making AI more inclusive for Assamese speakers."*

## 🎙️ 3. Project Workflow – Trishita (1.5 min)
*"Our project follows a systematic workflow, broken down into four key phases:

1️⃣ Dataset Collection & Preprocessing – We sourced Assamese text data from Wikipedia, online articles, and public datasets. We then cleaned, tokenized, and formatted it for training.

2️⃣ Model Selection & Training Setup – We selected Mistral-7B, a powerful open-source language model, and optimized it for Assamese chat completion using parameter-efficient fine-tuning (QLoRA).

3️⃣ Training & Optimization – Using Google Colab’s free GPU resources, we fine-tuned the model on our curated dataset while ensuring minimal computational cost.

4️⃣ Integration with Frontend – We have already developed a chat interface where users can interact with the model. Once training is complete, we will integrate the fine-tuned model into the system."*

## 🎙️ 4. Data Preparation & Training Process – Changrung (2 min)
*"Data preparation is a crucial step in building any language model. Assamese presents unique challenges, such as script variations and inconsistent formatting in online text sources.

To address this, we followed a multi-step approach:

Data Cleaning – We removed unnecessary symbols, numbers, and non-Assamese words to ensure high-quality input.
Tokenization & Encoding – We converted the text into a format that the model can process efficiently.
Dataset Splitting – We divided the data into 80% training and 20% validation to ensure balanced learning.
For training, we are using the Unsloth library for efficient fine-tuning on limited GPU resources. By leveraging QLoRA (Quantized Low-Rank Adaptation), we reduce memory usage while maintaining model accuracy. This ensures we get the best possible results within our hardware constraints."*

## 🎙️ 5. Challenges & Solutions – Rishikesh (1.5 min)
*"Every AI project comes with challenges, and NativeGPT was no exception. We faced three key challenges:

🔹 Limited Computational Resources – Training a large model like Mistral-7B requires high-end GPUs, but we only have access to a 4GB VRAM GPU. To overcome this, we used Google Colab’s free GPUs and optimized memory usage using QLoRA.

🔹 Data Scarcity & Quality Issues – Assamese text datasets are limited and often noisy. We manually curated and cleaned the data to improve quality.

🔹 Model Performance Tuning – Ensuring that the model generates fluent and accurate Assamese text was a challenge. We are iterating through multiple training runs, tweaking hyperparameters, and using a reinforcement-based approach to enhance output quality."*

## 🎙️ 6. Current Progress & Live Demo (if possible) – Tabarakul (Lead Presenter) (1.5 min)
*"So far, we have completed data collection, preprocessing, and model setup. Our fine-tuning process is in progress, and once complete, we will integrate it with our chat interface.

We have also built a fully functional frontend chat app where users will soon be able to interact with our model. While the final model is still training, we can demonstrate the working chat interface using an existing AI model as a placeholder."*

🚀 (If a live demo is possible, show the chat interface.)

## 🎙️ 7. Future Scope & Conclusion – Uday (1.5 min)
*"Our vision doesn’t stop at Assamese. In the future, we aim to:

✅ Expand to more Indian languages, making AI accessible in multiple regional dialects.
✅ Implement speech-to-text capabilities, allowing voice-based interactions.
✅ Deploy the model on cloud servers, making it scalable and accessible for businesses and developers.

With these improvements, NativeGPT can become a cornerstone for AI-driven Assamese applications in customer service, education, and digital communication.

Thank you for your time! We are happy to answer any questions."*

🎤 Q&A Session (Team Answers)

⏳ Total Time Breakdown
✔️ Introduction – 1.5 min
✔️ Problem Statement & Motivation – 2 min
✔️ Project Workflow – 1.5 min
✔️ Data Preparation & Training – 2 min
✔️ Challenges & Solutions – 1.5 min
✔️ Current Progress & Demo – 1.5 min
✔️ Future Scope & Conclusion – 1.5 min
✔️ Q&A – Remaining time

⏳ Total Duration: ~10 minutes