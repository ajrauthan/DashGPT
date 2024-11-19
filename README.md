---

### Framework Overview

This framework operates on the principles of **Retrieval-Augmented Generation (RAG)**, enabling seamless interaction with a database through a user-friendly **Dash Application**. It allows users to query the database, receive concise answer summaries, and view dynamic visualizations as output.

### How to Use

1. Clone the repository
2. Set up a virtual environment:
   - `python -m venv venv` (This creates a virtual environment named venv in your project folder)
   - `source venv/bin/activate` (This activates the virtual environment)
3. Upgrade pip: `pip install --upgrade pip`
4. Install the Requirements: `pip install -r requirements.txt`
5. Get the Google Cloud key:
   - Go to the Google Cloud Console: https://console.cloud.google.com/
   - Navigate to IAM & Admin > Service Accounts.
   - Create a new service account (or use an existing one).
   - Add Bigquery User & Admin roles to the service account.
   - Click Keys > Add Key > JSON to download the file.
   - Call this key `google-cloud.json` and copy it to the folder
6. Get the Open API key:
   - Log in to your OpenAI account at https://platform.openai.com/api-keys.
   - Navigate to API Keys.
   - Create a new API key if you don’t already have one.
   - Create a JSON file name open_api_creds.json with a structure like below and update your API key:
```
{
  "api_key": "your-openai-api-key"
}
```
7. Run the script: `python Dash_app_BQ_genGraph.py`
8. A link will be generated upon execution—click on it to access the Dash interface and start using the application.

---
