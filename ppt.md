## NativeGPT: Fine-Tuning a Language Model for Assamese Chat Completion

### 1. Project Overview

- **Objective:** Build a conversational AI model capable of chat completion in the Assamese language.
- **Approach:** Fine-tune a pre-trained model (Mistral-7B) using Unsloth for fast and efficient training.
- **End Goal:** Seamlessly integrate the model into a chat app for Assamese users.

### 2. Motivation

- Limited availability of AI models capable of understanding Assamese.
- High demand for local language support in chat-based applications.
- Potential for expanding local accessibility and language inclusivity.

### 3. Dataset Preparation

**Step 1:** Data Collection

- Gathered raw Assamese text data from web sources, newspapers, books, and social platforms.

**Step 2:** Data Cleaning

- Removed noise, special characters, and non-Assamese text.
- Ensured all data was in UTF-8 format for compatibility.

**Step 3:** Data Splitting

- Split data into:
  - **Train:** 80% (\~10,000 sentences)
  - **Validation:** 20% (\~2,000 sentences)

### 4. Model Selection

- **Selected Model:** Mistral-7B-v0.1 (open-source)
- **Why Mistral-7B?**
  - Lightweight and optimized for chat completion.
  - Easy to fine-tune with Unsloth.

### 5. Training Setup

**Training Framework:** Unsloth + Transformers\
**Hardware Setup:**

- **Primary:** Google Colab (for initial training)
- **Secondary:** Local GPU (NVIDIA GTX 1650 with 4GB VRAM)

**Training Parameters:**

- **Batch Size:** 2
- **Gradient Accumulation:** 4
- **Learning Rate:** 5e-5
- **Epochs:** 3

### 6. Tokenization

- Used Huggingface Tokenizer to tokenize Assamese text.
- Set maximum token length to 128.
- Converted text to token IDs for efficient training.

### 7. Training Progress

- ✅ **Data Cleaning & Preparation Completed**
- ✅ **Model Selection Finalized**
- ✅ **Training Setup Ready**
- 🔲 **Model Training in Progress (starting soon)**
- 🔲 **Deployment & Testing (future step)**

### 8. Frontend Chat App

- **Developed a beautiful chat interface** using React + Tailwind CSS.
- **Features:**
  - Prompt input for user.
  - AI-generated response in Assamese.
  - Clean UI/UX with fast response time.
- **Integration Plan:**
  - Once the model is trained, connect it via REST API.

### 9. Challenges Faced

- **Dataset Quality:** Initially struggled to find high-quality Assamese text data.
- **Hardware Limitations:** Training on GTX 1650 is slower but feasible.
- **Training Stability:** Overcoming Out-of-Memory (OOM) errors by using gradient accumulation.

### 10. Timeline & Progress

| Task                     | Status         |
| ------------------------ | -------------- |
| Data Collection          | ✅ Completed    |
| Data Cleaning            | ✅ Completed    |
| Model Selection          | ✅ Completed    |
| Fine-Tuning Setup        | ✅ Completed    |
| Model Training           | 🔲 In Progress |
| Frontend Chat App        | ✅ Completed    |
| Integration & Deployment | 🔲 Pending     |

### 11. Future Scope

- ✅ **Expand to Manipuri language** after successful Assamese deployment.
- ✅ **Deploy the model via REST API** for public use.
- ✅ **Optimize the model** to reduce latency and improve response time.

### 12. Possible Questions from Panel

**Q1:** Why did you choose Mistral-7B and not a larger model like LLaMA-13B?\
**Answer:** Mistral-7B provides a balance between performance and resource utilization. Given our hardware (1650 GPU), it was the most optimal choice.

**Q2:** How did you handle the lack of Assamese datasets?\
**Answer:** We manually sourced text data from public domains, newspapers, and Assamese literature. Additionally, we plan to release this dataset publicly for future research.

**Q3:** What challenges did you face during training?\
**Answer:** The primary challenges were:

- Handling low GPU memory.
- Cleaning and tokenizing text data.
- Ensuring training stability without interruptions.

**Q4:** What’s your plan after training is done?\
**Answer:** Once the model is trained, we will deploy it as a REST API and integrate it with our frontend chat app.

### 13. Conclusion

- Our project, **NativeGPT**, aims to break the language barrier by bringing AI chat models to underrepresented languages like Assamese.
- In the coming weeks, we will complete training, deploy the model, and showcase a working chat assistant in Assamese.

