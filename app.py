from fastapi import FastAPI, UploadFile, File
import shutil
import os
from inference import load_model, predict

app = FastAPI()

# Load the model once when the server starts
# Ensure your model file 'resnext50_cctv_anomaly.pth' is in the same directory
model = load_model("./resnext50_cctv_anomaly.pth")

@app.post("/predict_video")
async def predict_video(video: UploadFile = File(...)):
    # Save the uploaded video to a temporary file
    temp_video_path = f"temp_{video.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    try:
        # Run your existing predict function
        result = predict(temp_video_path, model)
    finally:
        # Clean up the temporary video file to save space
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    return result

@app.get("/")
def read_root():
    return {"message": "Video Anomaly Detection API is running!"}
