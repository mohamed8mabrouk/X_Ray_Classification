from pydantic import BaseModel, ConfigDict

class PredictionResponse(BaseModel):
    label: str
    
    processing_time_ms: float

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str # "ok" or "ready"
    model_loaded: bool # true if model is loaded
