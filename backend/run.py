"""Starts the KeyMic backend. Binds to localhost only by default."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "keymic.api:app",
        host=os.environ.get("KEYMIC_HOST", "127.0.0.1"),
        port=int(os.environ.get("KEYMIC_PORT", "8000")),
    )
