# Iris Classification API — Practical Task 6

Random Forest classifier trained on the built-in scikit-learn Iris dataset (150 samples, 4 features, 3 classes). Stratified 80/20 train/test split, random seed 42. Measurements are in centimeters.

## Local setup

Use Python 3.13. Create and activate a virtual environment, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python -m uvicorn main:app --reload
```

Open http://localhost:8000/ and http://localhost:8000/docs. In Swagger, expand POST /predict, select Try it out, enter the JSON below, then Execute.

```json
{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
```

Expected: `{"prediction":"Iris Setosa","class_id":0}`.

## Docker

Stop the local server before running the container on the same port. Start Docker Desktop, then run:

```bash
docker build -t iris-ml-api .
docker run --rm -p 8000:8000 --name iris-api iris-ml-api
```

Repeat the root and Swagger tests at the same URLs. Stop with Ctrl+C.

## Verification

See test-results.txt for actual local Uvicorn HTTP checks. Docker build/run was not verified in the preparation environment because Docker was unavailable.

## Submission

Create an empty GitHub repository, then run:

```bash
git init
git add .
git commit -m "Complete Practical Task 6"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ml-fastapi-docker.git
git push -u origin main
```

Replace YOUR_USERNAME with your GitHub username. Submit the repository URL.
