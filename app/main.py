from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from io import BytesIO
import os
import shutil
import mimetypes
from app.model import load_img, load_vid, BASE_DIR

app = FastAPI()

STATIC = os.path.join(BASE_DIR, 'static')

# Mount the "static" directory to serve static files (CSS, JS, images, etc.)
app.mount('/app/app/static', StaticFiles(directory=STATIC), name="static")

# Directory for uploads
UPLOAD_DIR = os.path.join(STATIC, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the HTML template from the "templates" folder
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded file
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    # Get the MIME type of the file to detect if it's an image or video
    mime_type, _ = mimetypes.guess_type(file_location)

    # Prepare to store predictions and output file info
    predictions = []
    output_file_path = None

    if mime_type.startswith('image'):
        # Process the image using your YOLO model
        # Assuming load_img returns a dictionary with predictions
        result_info = load_img(file_location)
        if result_info is None:
            # Return an error message if the video processing failed
            return JSONResponse(content={"error": "Video processing failed. Output video was not created."}, status_code=500)
        predictions = result_info["predictions"]
        output_file_path = result_info["out_path"]

    elif mime_type.startswith('video'):
        # Process the video and save the output video
        result_info = load_vid(file_location)
        if result_info is None:
            # Return an error message if the video processing failed
            return JSONResponse(content={"error": "Video processing failed. Output video was not created."}, status_code=500)
        predictions = result_info["predictions"]
        output_file_path = result_info["out_path"]

    # Render the result in predict.html with all the info passed to the template
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "file_path": output_file_path,
        "file_type": mime_type,
        "predictions": predictions
    })
