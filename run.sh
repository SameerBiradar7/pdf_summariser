#!/usr/bin/env bash
conda activate rag_flask_env
export FLASK_APP=app.app
export FLASK_ENV=development
flask run --host=127.0.0.1 --port=5000
