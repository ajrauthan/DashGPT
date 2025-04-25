---

### Framework Overview

This framework operates on the principles of **Retrieval-Augmented Generation (RAG)**, enabling seamless interaction with a database through a user-friendly **Dash Application**. It allows users to query the database, receive concise answer summaries, and view dynamic visualizations as output.

### Configuration

The application can be customized through the `config.yaml` file in the main directory. This file contains all the configurable settings:

```yaml
# Example configuration options
openai:
  model: "gpt-4.1"        # Choose your OpenAI model
  max_tokens: 1000        # Adjust token limit
  temperature: 0.2        # Control response randomness

bigquery:
  project_id: "your-project"      # Your BigQuery project
  dataset_id: "your-dataset"      # Your dataset name

app:
  debug: true            # Enable/disable debug mode
  port: 8050            # Change application port
  host: "127.0.0.1"     # Change host address
```

Modify these settings according to your needs before running the application. The application uses the BigQuery Storage API for optimized data fetching performance.

### How to Use

1. Clone the repository
2. Set up a virtual environment:
   - `python -m venv dashgpt` (This creates a virtual environment named dashgpt in your project folder)
   - `source dashgpt/bin/activate` (This activates the virtual environment)
3. Upgrade pip: `pip install --upgrade pip`
4. Install the Requirements: `pip install -r requirements.txt`
   - This includes the BigQuery Storage client for improved query performance
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
   - Create a new API key if you don't already have one.
   - Create a JSON file name open_api_creds.json with a structure like below and update your API key:
```
{
  "api_key": "your-openai-api-key"
}
```
7. Configure the application by editing `config.yaml` according to your needs
8. Run the script: `python Dash_app_BQ_genGraph.py`
9. A link will be generated upon execution—click on it to access the Dash interface and start using the application.

### Project Structure

```
.
├── config.yaml                   # Main configuration file
├── Dash_app_BQ_genGraph.py      # Main application file
├── components/                   # Components directory
│   ├── __init__.py
│   ├── chat_history_section.py  # Chat history component
│   ├── layout.py               # Main layout combining all components
│   ├── query_section.py        # Query input component
│   ├── summary_section.py      # Summary component
│   └── visualization_section.py # Visualization component
└── styles/                      # Styles directory
    ├── __init__.py
    └── theme.py                # Centralized styles
```

The project follows a modular structure where components and styles are separated into their own directories for better organization and maintainability. The main application file orchestrates these components while keeping the code clean and manageable.

---
