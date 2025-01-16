# AI Resume Enhancer

## Description

AI Resume Enhancer is a tool that leverages AI agents to provide a tailored resume based on a job post link and your actual resume. By analyzing the job post and your existing resume, the AI agents generate a customized resume that highlights the most relevant skills and experiences, increasing your chances of landing the job.

## Installation

### 1. Install Python 3.11

Ensure you have Python version 3.11 installed on your machine.

### 2. Create a Virtual Environment and Install Dependencies

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up Your Environment Variables

```sh
cp .env.example .env
```

Edit the `.env` file with your API keys.

### 4. Run the Resume Enhancer

```sh
python resume_enhancer.py
```

## Acknowledgements

This project was inspired by a course from [deeplearning.ai](https://www.deeplearning.ai/).
