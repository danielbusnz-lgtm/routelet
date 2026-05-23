"""FastAPI server exposing the trained router.

    POST /route  {"text": "..."}  ->  {"intent": "...", "confidence": 0.0}

The model loads once at startup and classifies on each request.
"""

# TODO: load the trained model on startup, classify the incoming text per request.
