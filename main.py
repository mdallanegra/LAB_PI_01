from fastapi import FastAPI

app = FastAPI()


@app.get("/UserForGenre")
def UserForGenre():
    result = ('Usuario con mas horas jugadas para genero')
    return result
