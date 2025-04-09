from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Initialize FastAPI
app = FastAPI(
    title="Clinical NER API using Biomedical-NER-All",
    description="NER using a pre-trained biomedical NER model that extracts clinical entities such as symptoms, lab findings, medications, diseases, vital signs, injuries, and vaccines.",
    version="2.1.0"
)

# Enable CORS for cross-origin requests if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load pre-trained biomedical NER model from Hugging Face.
# d4data/biomedical-ner-all has been trained on several biomedical datasets, 
# making it a strong off-the-shelf candidate to extract clinical entities.
MODEL_NAME = "d4data/biomedical-ner-all"  # No fine-tuning required
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
# Use aggregation strategy to merge sub-token predictions
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

print("Biomedical NER model loaded.")

# Pydantic models for request/response
class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    umls_code: Optional[str] = None

class ExtractionResponse(BaseModel):
    entities: List[Entity]

class GroupedExtractionResponse(BaseModel):
    groups: Dict[str, List[Entity]]

class TextInput(BaseModel):
    text: str

# Core function to extract entities using the NER pipeline
def extract_entities(text: str, threshold: float = 0.6) -> List[Entity]:
    raw_preds = ner_pipeline(text)
    results = []
    for pred in raw_preds:
        if pred.get("score", 1) >= threshold:
            results.append(Entity(
                text=pred["word"],
                type=pred["entity_group"].upper(),  # Label from the model
                start=pred["start"],
                end=pred["end"],
                umls_code=None
            ))
    return results

# Helper function to group entities by their type
def group_entities(entities: List[Entity]) -> Dict[str, List[Entity]]:
    groups = defaultdict(list)
    for entity in entities:
        groups[entity.type].append(entity)
    return dict(groups)

# Routes

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint that returns a flat list of extracted entities
@app.post("/extract", response_model=ExtractionResponse)
async def extract_medical_entities(input: TextInput):
    try:
        entities = extract_entities(input.text)
        return ExtractionResponse(entities=entities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint that returns extracted entities grouped by type
@app.post("/extract_grouped", response_model=GroupedExtractionResponse)
async def extract_medical_entities_grouped(input: TextInput):
    try:
        entities = extract_entities(input.text)
        groups = group_entities(entities)
        return GroupedExtractionResponse(groups=groups)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
