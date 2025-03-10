## Slide 1: Title Slide (Spoken by Tabarakul)
#### Content:
### Title: "NativeGPT – Building an Assamese Chat AI"
#### Subtitle: "A Language Model for Assamese with Mistral-7B and QLoRA"
- Team Members:
    - Tabarakul Islam Hazarika
    - Uday Sankor Mohan
    - Trishita Das
    - Changrung Modhur
    - Rishikesh Sonowal
- Institution: Kaziranga University, Department of Computer Science & Engineering
- Date: [Presentation Date]
- Add a simple, aesthetic background image (Assamese culture/tech AI fusion).
## Slide 2: Introduction (Spoken by Tabarakul)
#### Content:
- NativeGPT – What is it?
    - A chatbot designed to understand and generate Assamese text using Mistral-7B with QLoRA fine-tuning.
- Why are we building it?
    - Lack of AI tools for Assamese → Google & OpenAI have minimal support.
    - Assamese is spoken by 30M+ people → Needs better AI accessibility.
    - Goal: Help students, researchers, and general users interact with AI in Assamese.
- Visuals:
    - A simple graphic showing AI models (GPT, Gemini) vs. NativeGPT for Assamese.

## Slide 3: Problem Statement (Spoken by Uday)
#### Content:
- Current Challenges:
    - Assamese lacks high-quality NLP datasets → Hard to train AI.
    - Existing models (GPT, Gemini) struggle with Assamese grammar & dialects.
    - Low-resource AI training → Need for optimization on limited hardware.
- Our Solution:
    - Create an AI specifically trained for Assamese conversation.
    - Fine-tune an open-source LLM (Mistral-7B) with custom datasets.
    - Optimize for low-GPU hardware (QLoRA technique).
- Visuals:
    - A contrast table: Existing AI (Poor Assamese Support) vs. NativeGPT (Fluent & Accurate).
## Slide 4: Project Objectives (Spoken by Uday)
#### Content:
- Key Goals of NativeGPT:
    - ✅ Develop an Assamese language chatbot using Mistral-7B
    - ✅ Use QLoRA fine-tuning to optimize for low-GPU devices
    - ✅ Collect & preprocess high-quality Assamese text data
    - ✅ Build a frontend chat UI for easy interaction
    - ✅ Overcome data scarcity & hardware limitations
- Visuals:
    - A flowchart of objectives (Data → Model Training → Optimization → Chatbot Deployment).
## Slide 5: Workflow Diagram (Spoken by Trishita)
#### Content:
- How NativeGPT Works (4 Phases):
    - 1️⃣ Data Collection & Cleaning (Books, articles, social media)
    - 2️⃣ Model Training (Fine-tuning Mistral-7B)
    - 3️⃣ Optimization (QLoRA for efficiency)
    - 4️⃣ Frontend Development (Chatbot UI)
- Visuals:
    - A clear block diagram showing data processing → model training → chat deployment.
## Slide 6: Dataset Collection & Preprocessing (Spoken by Changrung)
#### Content:
- Sources of Data:
    - Assamese books, articles, Wikipedia, government documents.
    - Scraped social media posts & news reports (cleaned & filtered).
- Data Preprocessing Steps:
    - Tokenization (breaking text into words)
    - Filtering noise (removing duplicate or irrelevant text)
    - Creating prompts & responses for chatbot training
- Visuals:
    - A before-after table (Raw text vs. Cleaned text).
## Slide 7: Model Selection & Training Setup (Spoken by Changrung)
#### Content:
- Why Mistral-7B?
    - Lightweight yet powerful, open-source, efficient for low-GPU devices.
- Training Setup:
    - QLoRA fine-tuning → Uses low VRAM (4GB GPU optimization).
    - Training on our dataset (custom Assamese text).
- Visuals:
    - A comparison of Mistral-7B vs. GPT-3.5 (Assamese performance).
## Slide 8: Training Challenges & Solutions (Spoken by Rishikesh)
#### Content:
- Challenges Faced:
    - 🚨 Limited GPU resources (NVIDIA 1650, 4GB VRAM).
    - 🚨 Lack of clean Assamese datasets → Manual curation needed.
    - 🚨 Training time is high → Needed efficiency tricks.
- Solutions Implemented:
    - ✅ QLoRA compression → 4x less VRAM usage.
    - ✅ Using Google Colab Pro for training.
    - ✅ Dataset filtering & augmentation for better quality.
- Visuals:
    - Screenshot of training logs & GPU utilization.
## Slide 9: Current Progress & Chat Interface (Spoken by Tabarakul)
#### Content:
- Backend: Model training almost complete.
- Frontend: Chat interface built with React + Flask.
- Features Implemented:
    - Real-time message streaming
    - Dark & Light Mode UI
    - User-friendly chat history storage
- Visuals:
Screenshots of the chat app UI.
## Slide 10: Live Demo (Optional, Spoken by Tabarakul)
- If model is ready, show a live chatbot interaction on-screen.
## Slide 11: Future Scope & Improvements (Spoken by Uday)
#### Content:
- Upcoming Enhancements:
    - 🚀 Add voice-based chatbot using ElevenLabs AI.
    - 🚀 Deploy model on cloud servers for accessibility.
    - 🚀 Extend support to other regional languages (Bodo, Manipuri).
- Visuals:
    - A roadmap timeline showing future milestones.
## Slide 12: Conclusion (Spoken by Uday)
#### Content:
- Summary of Work Done:
    - ✅ Built Assamese language chatbot.
    - ✅ Overcame dataset & GPU challenges.
    - ✅ Developed a user-friendly chat UI.
    - ✅ Future improvements in voice AI & cloud deployment.
- Final Statement:
    - "NativeGPT is our step towards bridging the AI gap for Assamese speakers!"
## Slide 13: Q&A (All Members)
- Keep this ## Slide simple:
"Thank You! Any Questions?"