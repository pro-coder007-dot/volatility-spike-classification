from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import PredictRequest, PredictResponse, BatchPredictResponse
from app.model import predict_spike, model, features
from io import StringIO
import pandas as pd
import logging
from functools import wraps
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

def handle_prediction_errors(func):
    @wraps(func)
    async def wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f'Missing required field: {str(e)}')
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f'Invalid input: {str(e)}')
        except Exception as e:
            raise HTTPException(status_code=500, detail='Prediction failed')
    return wrapper

# endpoint for prediction
@app.post('/predict',response_model=PredictResponse)
@handle_prediction_errors
def predict(req: PredictRequest):
    spike, prob = predict_spike(req.model_dump()) # model_dump() -> dict of the request data 
    return PredictResponse(
        spike_predicted=spike, # from model.py after ensuring all the validations
        probability=prob
    )

@app.get('/health')
def health_check():
    return {'status': 'ok', 'model_loaded': True}

@app.get('/model/info')
def model_info():
    return {
        'model_type': type(model).__name__,
        'n_features': len(features),
        'features': features
    }

@app.get('/model/feature_importance')
def feature_importance():
    importances = model.feature_importances_
    ranked = sorted(
        zip(features, importances), # eg. zip(['open', 'close', 'volume'], [0.2, 0.5, 0.3])
        key=lambda x: x[1],
        reverse=True
    )
    return {
        'feature_importance': [
            {'feature': f, 'importance': round(i, 4)}
            for f, i in ranked
        ]
    }

@app.post('/predict/batch',response_model=list[PredictResponse])
def predict_batch(requests: list[PredictRequest]):
    results = []
    for req in requests:
        spike, prob = predict_spike(req.model_dump())
        results.append(PredictResponse(
            spike_predicted=spike,
            probability=prob
        ))
    return results

@app.post('/predict/csv', response_model=BatchPredictResponse)
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are allowed')
    
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to read CSV: {str(e)}')
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    required_cols = {'adj_close', 'close', 'high', 'low', 'open', 'volume',
                     'day_of_week', 'month', 'day', 'week_of_year', 'return', 'volatility'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f'Missing columns: {", ".join(missing_cols)}')
    
    predictions = []
    
    for idx, row in df.iterrows():
        try:
            data = row.to_dict()
            data['return_'] = data.pop('return')

            spike, prob = predict_spike(data)
            predictions.append(PredictResponse(
                spike_predicted=spike,
                probability=prob
            ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error processing row {idx}: {str(e)}')
        
    spike_count = sum(1 for p in predictions if p.spike_predicted)

    return BatchPredictResponse(
        total_rows=len(predictions),
        spike_count=spike_count,
        spike_percentage=round(spike_count / len(predictions) * 100, 2),
        predictions=predictions
    )