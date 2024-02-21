from fastapi import FastAPI, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from processing import generate_response
import aiofiles

app = FastAPI()
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/")
async def main():
    async with aiofiles.open('templates/index.html', mode='r') as f:
        content = await f.read()
    return HTMLResponse(content)

@app.post("/")
async def translate(input_data: dict = Body(...)):
    user_input = input_data.get("inputValue")
    response = generate_response(user_input)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)