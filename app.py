import gradio as gr
import os
import shutil
from inference import load_model, predict

# Load model
model = load_model("./resnext50_cctv_anomaly.pth")

def predict_video_gradio(video_path):
    if video_path is None:
        return "Please upload a video."
    
    # Run the existing predict function
    result = predict(video_path, model)
    
    # Format the result nicely for the Gradio UI
    output_text = f"**Prediction:** {result['predicted_class'].upper()}\n"
    output_text += f"**Confidence:** {result['confidence']}%\n\n"
    output_text += "**Top 3 Predictions:**\n"
    
    for r in result["top_k"]:
        output_text += f"- {r['class']}: {r['confidence']}%\n"
        
    return output_text

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_video_gradio,
    inputs=gr.Video(label="Upload CCTV Video"),
    outputs=gr.Markdown(label="Detection Results"),
    title="CCTV Video Anomaly Detection",
    description="Upload a short video clip to detect anomalies such as assault, fighting, robbery, shoplifting, stealing, vandalism, etc. The model processes 16 frames to classify the activity.",
    examples=[],
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
