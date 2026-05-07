from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
 
 
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = {
        "title": "Next Word AI"
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)





if __name__ == "__main__":   
    uvicorn.run(app, host="127.0.0.1", port=8000)