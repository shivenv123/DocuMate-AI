from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import traceback

# Initialize FastAPI
app = FastAPI(
    title="Clinical NER API",
    description="NER using d4data/biomedical-ner-all",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variable to track if model is loaded
model_loaded = False
ner_pipeline = None

# Pydantic models for request/response
class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    umls_code: Optional[str] = None

class ExtractionResponse(BaseModel):
    entities: List[Entity]

class TextInput(BaseModel):
    text: str

# Function to load model (called when needed)
def load_model():
    global ner_pipeline, model_loaded
    if not model_loaded:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            MODEL_NAME = "d4data/biomedical-ner-all"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple"
            )
            model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
            raise e

# Core function to extract entities
def extract_entities(text: str) -> List[Entity]:
    # Make sure model is loaded
    if not model_loaded:
        load_model()
    
    raw_preds = ner_pipeline(text)
    results = []
    for pred in raw_preds:
        results.append(Entity(
            text=pred["word"],
            type=pred["entity_group"].upper(),  # e.g., DISEASE, CHEMICAL, GENE
            start=pred["start"],
            end=pred["end"],
            umls_code=None  # UMLS linking not included in this model
        ))
    return results

# Routes
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/extract", response_model=ExtractionResponse)
async def extract_medical_entities(input: TextInput):
    try:
        entities = extract_entities(input.text)
        return ExtractionResponse(entities=entities)
    except Exception as e:
        print(f"Error processing text: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Try to load the model at startup
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {e}")
        print("Model will be loaded when needed.")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)