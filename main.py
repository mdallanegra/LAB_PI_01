from fastapi import FastAPI

app = FastAPI()


@app.get("/read_root")
def read_root():
    return "Welcome to render"
