from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from main import ImageProcessor
import numpy as np
import cv2
import random

# cors
origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# post request for image
@app.post("/image")
async def image_endpoint(image: UploadFile = File(...)):
    try:
        # convert image to numpy array
        image = await image.read()
        image = np.asarray(bytearray(image), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return {"status": True, "message": ImageProcessor().process_image(image)}
    except Exception:
        return {"status": False,"message": "Something went wrong"}

@app.post("/pdf")
async def pdf_endpoint(pdf: UploadFile = File(...)):
    # save pdf to disk as a temporary file
    
    filename = f"./uploads/upload_{random.randint(0, 100000000)}.pdf"
    with open(filename, "wb") as f:
        f.write(await pdf.read())
    
    return {"status": True, "message": ImageProcessor().process_pdf(filename)}

@app.post("/create_csv")
async def create_csv_endpoint(file: UploadFile = File(...)):
    # save pdf to disk as a temporary file
    if file.filename.endswith(".pdf"):
        filetype = "pdf"
        filename = f"./uploads/upload_{random.randint(0, 100000000)}.pdf"
        with open(filename, "wb") as f:
            f.write(await file.read())
        return {"status": True, "message": ImageProcessor().create_csv(filetype=filetype, filepath=filename)}
    elif file.filename.endswith(".jpg") or file.filename.endswith(".png"):
        filetype = "image"
        image = await file.read()
        image = np.asarray(bytearray(image), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return {"status": True, "message": ImageProcessor().create_csv(filetype=filetype, image=image)}
    else:
        return {"status": False, "message": "Invalid filetype"}
    