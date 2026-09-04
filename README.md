# Diabetes Risk Check — deployment guide (Render, free)

## What's in this folder
- `app.py` — Flask backend. Loads `model/XGBoost.pkl` and exposes:
  - `GET /` → serves the intake page
  - `POST /predict` → takes the 21 features as JSON, returns a simplified risk result
- `templates/index.html` — the step-by-step intake wizard, wired to call `/predict`
- `model/XGBoost.pkl` — your trained model
- `requirements.txt`, `Procfile` — deployment config

## How the result is simplified
The model itself predicts 3 classes (no diabetes / prediabetes / diabetes). The backend
merges "prediabetes" and "diabetes" probabilities into a single **"at risk"** score, and
shows the user either **"Low risk right now"** or **"You may be at risk"**, along with the
underlying percentages. If you ever want to show all 3 categories instead, that logic lives
in the `/predict` route in `app.py` — straightforward to change.

## Deploy to Render (free tier)

1. **Put this folder in a GitHub repo.**
   - Create a new repo on GitHub, then from this folder:
     ```
     git init
     git add .
     git commit -m "Diabetes risk check app"
     git branch -M main
     git remote add origin https://github.com/<your-username>/<repo-name>.git
     git push -u origin main
     ```

2. **Create the Render service.**
   - Go to https://render.com and sign up / log in (free).
   - Click **New +** → **Web Service**.
   - Connect your GitHub account and select the repo you just pushed.

3. **Configure the service.**
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
   - Click **Create Web Service**.

4. **Wait for the build to finish.** Render will install dependencies and start the app.
   You'll get a live URL like `https://your-app-name.onrender.com`.

5. **Note on the free tier:** the service goes to sleep after ~15 minutes of no traffic and
   takes 30–60 seconds to wake back up on the next visit. This is normal for Render's free
   tier and not a bug in the app.

## Running it locally first (optional but recommended)
```
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

## Updating the model later
Drop a new `.pkl` into `model/`, update `MODEL_PATH` in `app.py` if you rename the file,
and make sure `FEATURE_ORDER` in `app.py` still matches `model.feature_names_in_` for the
new model — mismatched feature order will silently produce wrong predictions.
