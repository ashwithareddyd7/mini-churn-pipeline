python ingest.py
python transform.py
python analytics.py
python feature_store.py
python train.py

echo "Pipeline complete."
echo "Start API server: uvicorn serve:app --reload --port 8000"
echo "Start Streamlit UI: streamlit run streamlit_app.py"