from pydantic import BaseModel, field_validator
import math

class PredictRequest(BaseModel):
    adj_close:    float
    close:        float
    high:         float
    low:          float
    open:         float
    volume:       float
    day_of_week:  int
    month:        int
    day:          int
    week_of_year: int
    return_:      float
    volatility:   float

@field_validator('adj_close', 'high', 'low', 'open', 'close')
def prices_must_be_positive(cls, v):
    if v <= 0:
        raise ValueError('Price values must be positive')
    return v

@field_validator('volume')
def volume_must_be_positive(cls, v):
    if v <= 0:
        raise ValueError('Volume must be positive')
    return v

@field_validator('volatility', 'return_')
def must_be_finite(cls, v):
    if not math.isfinite(v):
        raise ValueError('Volatility and return must be finite numbers')
    return v

@field_validator('day_of_week')
def day_of_week_must_be_valid(cls, v):
    if not 0 <= v <= 6:
        raise ValueError('day_of_week must be between 0 (Monday) and 6 (Sunday)')
    return v

@field_validator('month')
def month_must_be_valid(cls, v):
    if not 1 <= v <= 12:
        raise ValueError('month must be between 1 and 12')
    return v

class PredictResponse(BaseModel):
    spike_predicted: bool
    probability: float

# Validation for response for file
class BatchPredictResponse(BaseModel):
    total_rows: int
    spike_count: int
    spike_percentage: float
    predictions: list[PredictResponse]

