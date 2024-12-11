from fastapi import FastAPI
import uvicorn
from src.config import get_log_config, get_dev_mode, get_num_workers

app = FastAPI()


@app.get("/index")
def hello_world():
    return "Hello world"


if __name__ == "__main__":
    uvicorn.run(
        "digital_assistant_rag.app:app",
        host="0.0.0.0",
        port=5000,
        log_config=get_log_config(),
        reload=get_dev_mode(),
        workers=get_num_workers(),
    )
