# retinalAI

RetinalAI is a Flask web application for detecting diabetic retinopathy (DR) from retinal fundus images and for basic patient record management and analytics. The app includes a pretrained model file (EfficientNetB6_Expert_Optimized.h5) and provides dashboards, history, and admin features.

## Repository contents (important files)

- `app.py` — Flask application and routes (entry point).
- `config.py` — Application configuration (UPLOAD_FOLDER, DB URI, secret key, mail settings).
- `models.py` — SQLAlchemy models and DB setup.
- `utils.py` — Prediction helpers (loads the model and utilities such as `predict_dr_stage`).
- `EfficientNetB6_Expert_Optimized.h5` — Pretrained model used by `utils.predict_dr_stage` (should remain in repo root or the path expected by `utils.py`).
- `requirements.txt` — Python dependencies.
- `run_app.sh` — Convenience script to run the app with the project's recommended Python virtualenv.
- `static/uploads/` — Folder where uploaded images are stored (configured via `UPLOAD_FOLDER`).
- `data/sample_dr_cases.csv` — Sample external dataset used by the external analytics feature.

## Quick overview (what I found in `app.py`)

- App creation and configuration are near the top of `app.py`:

  - Flask app is created and configured with `app = Flask(__name__)` and `app.config.from_object(Config)` (reads `config.Config`).

  - Database initialization: `db.init_app(app)` and `db.create_all()` are executed on startup.

  - A default admin user is created automatically (if missing) with username `sowkya` and password `1234` (see `app.py`).

  - Prediction flow: image upload -> `utils.validate_image` -> `predict_dr_stage(file_path)` -> `check_operation_suitability(...)` -> saved to DB.

  - App entrypoint (how the server is started) is at the bottom of `app.py`:

```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
```

  That means you can run the app directly with `python app.py` (or use the included `run_app.sh`).

## Environment & prerequisites

- Recommended Python: 3.11 (project includes `run_app.sh` that references `./.venv311/bin/python`).
- Install dependencies:

```bash
python -m venv .venv311   # or your preferred venv name
source .venv311/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` lists `tensorflow` without a pinned version — install a build appropriate for your platform (e.g. `pip install tensorflow==2.12.0` for macOS Intel/Apple Silicon as needed). If you encounter issues installing TensorFlow, refer to the official TensorFlow install guide.

## Required environment variables

- `SECRET_KEY` — optional (defaults to `dev-secret-key-12345` in `config.py`).
- `DATABASE_URL` — optional (defaults to `sqlite:///site.db`).
- `EMAIL_USER` and `EMAIL_PASS` — optional if you want mail features to work.
- `PORT` — optional (defaults to `10000`).

You can create a `.env` file and export these before running, or export them in your shell session:

```bash
export SECRET_KEY="your-secret"
export DATABASE_URL="sqlite:///instance/site.db"
export EMAIL_USER="you@example.com"
export EMAIL_PASS="your-email-password"
export PORT=10000
```

## Setup (first-time)

1. Create and activate a virtual environment (see commands above).
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure the uploads folder exists and is writable (matches `UPLOAD_FOLDER` in `config.py`):

```bash
mkdir -p static/uploads
chmod 755 static/uploads
```

4. Ensure the pretrained model `EfficientNetB6_Expert_Optimized.h5` is present in the repository root (or the path expected by `utils.py`).

5. Optionally, create an `instance/` folder to store the sqlite DB if you want `instance/site.db`:

```bash
mkdir -p instance
```

6. Start the app (see next section).

## Running the app (development)

Option A — with the convenience script (recommended if you used `.venv311`):

```bash
./run_app.sh
```

Option B — run directly with Python:

```bash
python app.py
```

After starting, open http://localhost:10000 in your browser (or the port you set via `PORT`).

The app creates the DB tables automatically and creates a default admin user if not present:

- username: `sowkya`
- password: `1234`

Change the admin password after first login via the Admin UI.

## Running in production

For production, use a WSGI server like `gunicorn` and put the app behind a reverse proxy. Example:

```bash
# inside activated venv
gunicorn --bind 0.0.0.0:10000 app:app
```

Make sure you set `SECRET_KEY` and configure `DATABASE_URL` for a production-grade DB (Postgres, MySQL, etc.).

## Notes & troubleshooting

- Model loading: `utils.predict_dr_stage` expects the model file and may load TensorFlow; if memory or install issues occur, ensure TensorFlow is the correct build for your macOS platform.
- Database: By default the app uses SQLite (`sqlite:///site.db`). `config.py` points to OS `getcwd()` for `UPLOAD_FOLDER` — if you run the app from a different CWD, adjust paths or set `UPLOAD_FOLDER` in environment and/or `config.py`.
- If you upload large images, `MAX_CONTENT_LENGTH` is set to 16MB in `config.py`.
- If you see errors about missing tables, the app runs `db.create_all()` on startup inside an app context — ensure the process has write permission to create the DB file.

## Helpful developer commands

Activate venv, then:

```bash
# Run tests or quick linting (if you add tests/lint)
# Start dev server
python app.py

# Or with gunicorn
gunicorn --workers 3 --bind 0.0.0.0:10000 app:app

# To inspect DB (sqlite)
sqlite3 instance/site.db 
```

## Next steps I can do for you

- Create a small `.env.example` with environment variables to commit.
- Add instructions to automatically create and activate the recommended venv inside `run_app.sh`.
- Add a short CONTRIBUTING or deployment guide.

If you want, I can add this `README.md` file to the repository now (I can also tweak wording or add images/screenshots). Tell me if you'd like any edits or extra sections (e.g., API docs, endpoint list, developer notes).

---

README generated from analysis of `app.py`, `config.py`, and `run_app.sh`.
