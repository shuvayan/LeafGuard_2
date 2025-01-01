import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to logo in assets folder
logo_path = os.path.join(current_dir, "assets", "logo.png")

import gradio as gr
from pathlib import Path
from leafguard_capstone.prediction_service.predictions import make_prediction
from leafguard_capstone.gradio_ui.llm_integration import LLMService,apk



# Initialize LLM service
llm_service = LLMService(apk)

# List of supported languages
languages = ["English", "हिंदी", "ಕನ್ನಡ", "मराठी", "తెలుగు", "മലയാളം", "ਪੰਜਾਬੀ", "ଓଡ଼ിଆ", "বাংলা"]

def process_image(image, language):
    try:
        if image is None:
            return "Please upload an image first."
            
        # Run prediction using the uploaded image
        results = make_prediction(image.name, apply_augmentation=True)
        disease_name = results['voting_prediction']
        
        # Format output
        output = "🔍 ANALYSIS RESULTS\n"
        output += "=" * 50 + "\n\n"
        output += "📊 PREDICTION\n"
        output += f"Detected Disease: {disease_name}\n"
        output += f"Confidence: {results['averaging_prediction']}\n\n"
        
        # Get treatment if disease detected
        if 'healthy' not in disease_name.lower():
            treatment = llm_service.get_disease_treatment(disease_name)
            output += "💊 TREATMENT INFORMATION\n"
            output += "=" * 50 + "\n"
            output += treatment
        else:
            output += "✅ Plant appears healthy! No treatment needed."
            
        return output
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_question(message, history):
    # For now, keep the basic chat response
    return f"Response to: {message}"

def text_to_speech(text):
    # Keep the placeholder for TTS logic
    return None

# Rest of your original UI code remains exactly the same
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Custom CSS for styling
    gr.HTML("""
        <style>
        .header {
            text-align: center;
            padding: 20px;
        }
        .logo {
            width: 80px;
            height: 80px;
        }
        .title {
            color: #4CAF50;
            margin-top: 10px;
        }
        .textbox-container {
            position: relative;
        }
        .tts-button {
            position: absolute;
            right: 10px;
            bottom: 10px;
        }
        </style>
    """)
    
    # Header with logo and title
    with gr.Row(elem_classes="header"):
        with gr.Column():
            gr.Image(logo_path, elem_classes="logo")
            gr.Markdown("# Leaf Guard", elem_classes="title")
            gr.Markdown("## Intelligent Plant Disease Detection System")
    
    # Language selector
    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=languages,
            value="English",
            label="Select Language",
            interactive=True
        )
    
    # File upload section
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="Upload Plant Image",
                file_types=["image"],  # Removed PDF
                file_count="single"
            )
            gr.Markdown("Supported formats: JPG, PNG, JPEG | Max file size: 250kb")
            submit_btn = gr.Button("Submit", variant="primary")
    
    # Observations section with TTS
    with gr.Row():
        with gr.Column(elem_classes="textbox-container"):
            observations = gr.Textbox(
                label="Observations",
                placeholder="Analysis results will appear here...",
                interactive=False,
                lines=10  # Increased for better visibility
            )
            audio_output1 = gr.Audio(visible=False)
            tts_btn1 = gr.Button("🔊", elem_classes="tts-button")
    
    # Chat interface with TTS
    with gr.Column():
        chatbot = gr.Chatbot(
            label="Discussion",
            height=300
        )
        with gr.Row():
            with gr.Column(scale=20, elem_classes="textbox-container"):
                msg = gr.Textbox(
                    label="Type any questions here",
                    placeholder="Ask about the analysis..."
                )
                audio_output2 = gr.Audio(visible=False)
                tts_btn2 = gr.Button("🔊", elem_classes="tts-button")
    
    # Event handlers - kept exactly the same
    submit_btn.click(
        fn=process_image,
        inputs=[file_input, language_dropdown],
        outputs=observations
    )
    
    msg.submit(
        fn=process_question,
        inputs=[msg, chatbot],
        outputs=[chatbot],
        queue=False
    )
    
    # TTS event handlers
    tts_btn1.click(
        fn=text_to_speech,
        inputs=[observations],
        outputs=[audio_output1]
    )
    
    tts_btn2.click(
        fn=text_to_speech,
        inputs=[msg],
        outputs=[audio_output2]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()