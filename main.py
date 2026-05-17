from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tokenizers import Tokenizer
from contextlib import asynccontextmanager

# Import the model architecture class from your separate file
from model import KeyboardTransformer

# Define global variables to hold our model components in memory
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: Load the model components once ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model on device: {device}...")

    # 1. Initialize the architecture template
    model = KeyboardTransformer()

    # 2. Inject the trained weights into the template
    # map_location ensures it safely loads regardless of the hardware
    model.load_state_dict(torch.load("keyboard_model.pth", map_location=device))
    model.to(device)
    model.eval()  # Put the model in production evaluation mode

    # 3. Load the Tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # Store them in our global lifespan dictionary
    ml_models["model"] = model
    ml_models["tokenizer"] = tokenizer
    ml_models["device"] = device
    ml_models["block_size"] = 32  # Your context window length

    print("Model and Tokenizer loaded successfully!")
    yield
    # --- SHUTDOWN: Clean up resources if necessary ---
    ml_models.clear()


# Initialize FastAPI with the lifecycle manager
app = FastAPI(lifespan=lifespan)

# Initialize Jinja2Templates to serve HTML
templates = Jinja2Templates(directory="templates")


# Define what the incoming request data should look like
class PredictionRequest(BaseModel):
    text: str
    top_k: int = 5  # Default to providing your 5 suggestions


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"title": "Next Word AI"})


@app.post("/predict")
async def predict_next_words(request: PredictionRequest):
    try:
        # Retrieve components from the lifespan memory storage
        model = ml_models["model"]
        tokenizer = ml_models["tokenizer"]
        device = ml_models["device"]
        block_size = ml_models["block_size"]

        # 1. Tokenize incoming user input text
        tokens = tokenizer.encode(request.text).ids

        # 2. Crop input to your exact context threshold (last 32 tokens)
        tokens = tokens[-block_size:]
        tokens_tensor = torch.tensor([tokens]).to(device)

        # 3. Run inference without tracking gradients (saves performance)
        with torch.no_grad():
            logits, _ = model(tokens_tensor)
            # Isolate the predictions for the final word token in the sequence
            last_word_logits = logits[0, -1, :]

            # Calculate probabilities and extract top matches
            probs = F.softmax(last_word_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, request.top_k)

        # 4. Decode numerical array elements back to human string array
        suggestions = []
        for i in range(request.top_k):
            word = tokenizer.decode([top_indices[i].item()])
            suggestions.append(word)

        return {
            "input": request.text,
            "suggestions": suggestions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":   
    uvicorn.run(app, host="127.0.0.1", port=8000)