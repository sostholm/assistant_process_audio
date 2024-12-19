import uvicorn
from assistant_process_audio import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)