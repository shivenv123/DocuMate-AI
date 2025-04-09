# Clinical NER API

A FastAPI-based REST API that performs Named Entity Recognition (NER) on clinical text using the Biomedical-NER-All model. This service extracts various medical entities such as symptoms, lab findings, medications, diseases, vital signs, injuries, and vaccines from input text.

## Features

- Extract medical entities from clinical text
- Group extracted entities by their types
- RESTful API endpoints
- Web interface for testing
- CORS support for cross-origin requests
- Health check endpoint

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone the repository:

```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required packages:

```bash
pip install fastapi uvicorn transformers torch medspacy
```

## Usage

### Starting the Server

Run the following command to start the API server:

```bash
python clinical-ner-api/main.py
```

The server will start on `http://127.0.0.1:8000`

### API Endpoints

1. **GET /** - Web interface for testing the API
2. **POST /extract** - Extract medical entities (returns flat list)

   - Request body: `{"text": "your clinical text here"}`
   - Returns: List of extracted entities with their positions and types

3. **POST /extract_grouped** - Extract and group medical entities by type

   - Request body: `{"text": "your clinical text here"}`
   - Returns: Entities grouped by their types

4. **GET /health** - Health check endpoint
   - Returns: `{"status": "healthy"}`

### Example Request

```python
import requests

url = "http://localhost:8000/extract"
data = {"text": "Patient presents with severe headache and fever of 101.5F"}

response = requests.post(url, json=data)
print(response.json())
```

## Response Format

### Flat List (/extract)

```json
{
  "entities": [
    {
      "text": "headache",
      "type": "SYMPTOM",
      "start": 24,
      "end": 32,
      "umls_code": null
    },
    {
      "text": "fever",
      "type": "SYMPTOM",
      "start": 37,
      "end": 42,
      "umls_code": null
    }
  ]
}
```

### Grouped (/extract_grouped)

```json
{
  "groups": {
    "SYMPTOM": [
      {
        "text": "headache",
        "type": "SYMPTOM",
        "start": 24,
        "end": 32,
        "umls_code": null
      },
      {
        "text": "fever",
        "type": "SYMPTOM",
        "start": 37,
        "end": 42,
        "umls_code": null
      }
    ]
  }
}
```

## Model Information

This API uses the `d4data/biomedical-ner-all` model from Hugging Face, which has been trained on various biomedical datasets. The model can identify multiple types of medical entities and is used without fine-tuning.

## Configuration

- Default host: 127.0.0.1
- Default port: 8000
- Entity confidence threshold: 0.6 (configurable)

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines]

## Support

[Your Support Information]
