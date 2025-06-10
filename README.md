# AI-Powered Expense Tracker via WhatsApp

This project is a sophisticated, AI-powered expense tracker that allows users to manage their finances by sending messages through WhatsApp. It leverages Natural Language Processing (NLP) to understand and process user messages, extract expense details, and store them in a MongoDB database. The application is built with Flask and integrates with Twilio for WhatsApp communication, Groq for large language model (LLM) capabilities, and Hugging Face for sentence similarity tasks.

## Features

- **Track Expenses via WhatsApp**: Add, query, and delete expenses by sending simple text messages or voice notes.
- **Natural Language Understanding**: Utilizes a Groq LLM to interpret a wide range of user messages for adding, querying, and deleting expenses.
- **Voice-to-Text Transcription**: Transcribes WhatsApp voice notes into text for processing using Groq's audio transcription.
- **Intelligent Category Mapping**:
    - **Vector-Based Categorization**: Employs a Hugging Face sentence-transformer model to dynamically map expense items to the most relevant canonical category based on semantic similarity.
    - **LLM Fallback**: If vector similarity is low, it uses a Groq LLM for more nuanced category mapping.
    - **Regex Fallback**: A legacy regex-based system ensures that expenses are categorized even if the primary methods fail.
- **Automatic Typo Correction**: Leverages a Groq LLM to correct spelling and grammatical errors in user messages, improving the accuracy of expense logging.
- **Flexible Date Parsing**: Automatically parses relative dates like "yesterday," "tomorrow," and "day before yesterday" into absolute dates.
- **MongoDB Integration**: Securely stores and retrieves user expense data using a MongoDB database.

## How It Works

1.  **Receive Message**: The Flask application receives an incoming WhatsApp message (text or voice) from a user via a Twilio webhook.
2.  **Transcribe (if audio)**: If the message is a voice note, it is sent to the Groq API for transcription.
3.  **Classify Intent**: The transcribed or original text is sent to a Groq LLM to classify the user's intent as `add`, `query`, or `delete`.
4.  **Extract Information**: Based on the intent, another LLM prompt extracts key details like the amount, item, date, and context.
5.  **Categorize Expense**:
    - The expense item is first compared against a list of canonical categories using a Hugging Face sentence similarity model.
    - If the similarity score is below a set threshold, a Groq LLM is used as a fallback to determine the best category.
6.  **Database Operation**: The processed expense data is used to perform the corresponding action (insert, find, or delete) in the MongoDB database.
7.  **Send Response**: A confirmation or the requested information is sent back to the user on WhatsApp.

## Tech Stack

- **Backend**: Flask
- **AI & NLP**:
    - Groq (LLM for intent classification, data extraction, and typo correction; audio transcription)
    - Hugging Face (Sentence similarity for category mapping)
- **Database**: MongoDB
- **Messaging**: Twilio API for WhatsApp
- **Libraries**: `groq`, `huggingface_hub`, `pymongo`, `flask`, `twilio`, `word2number`, `dateparser`

## Getting Started

### Prerequisites

- Python 3.7+
- A [MongoDB](https://www.mongodb.com/) database (local or cloud-hosted on a service like MongoDB Atlas)
- A [Twilio](https://www.twilio.com/) account with a WhatsApp-enabled number
- A [Groq](https://wow.groq.com/) API key
- A [Hugging Face](https://huggingface.co/) API key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root directory of the project and add the following environment variables:


-   `GROQ_API_KEY`: Your API key from Groq.
-   `HUGGINGFACE_API_KEY`: Your API key from Hugging Face.
-   `MONGO_URI`: Your MongoDB connection URI. For a local instance, this might be `"mongodb://localhost:27017"`.

### Running the Application

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```
    The application will run on `http://localhost:5000` by default.

2.  **Expose your local server to the internet:**
    Since the Twilio webhook needs a public URL, use a tool like [ngrok](https://ngrok.com/) to expose your local server.
    ```bash
    ngrok http 5000
    ```
    Note the public URL provided by ngrok (e.g., `https://your-ngrok-subdomain.ngrok.io`).

3.  **Configure the Twilio Webhook:**
    - Go to your Twilio console and navigate to your WhatsApp sandbox or sender settings.
    - In the "When a message comes in" field, set the webhook URL to your public ngrok URL followed by `/whatsapp`:
      `https://your-ngrok-subdomain.ngrok.io/whatsapp`

## Usage

You can now interact with the expense tracker by sending messages to your configured WhatsApp number.

### Examples

-   **Adding an expense:**
    -   `spent 250 rs on groceries`
    -   `15 dollars for a movie ticket yesterday`
    -   (Voice Note) "Paid fifty for coffee"

-   **Querying expenses:**
    -   `show all my expenses`
    -   `how much did I spend on food this month?`
    -   `list my travel expenses`

-   **Deleting an expense:**
    -   `delete expense 60c72b2f9b1e8b3b3c8b4567` (replace with a valid expense ID)

### API Endpoint

You can also interact with the tracker directly through the `/process_command` API endpoint.

**URL**: `http://localhost:5000/process_command`
**Method**: `POST`
**Body** (JSON):

```json
{
    "message": "spent 100 on lunch",
    "user_id": "api_user_123"
}
