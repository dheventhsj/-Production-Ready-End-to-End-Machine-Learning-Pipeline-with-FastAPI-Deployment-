# End-to-End ML Pipeline (Production Ready)

This project demonstrates a production-style ML pipeline:
- Data ingestion
- Preprocessing
- Model training
- Evaluation
- Model serving via FastAPI

## Run
pip install -r requirements.txt
python run_pipeline.py
uvicorn app.main:app --reload

Visit: http://127.0.0.1:8000/docs
