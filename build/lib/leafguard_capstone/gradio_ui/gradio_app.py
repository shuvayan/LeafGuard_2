import gradio as gr
import os

# List of supported languages
languages = ["English", "हिंदी", "ಕನ್ನಡ", "मराठी", "తెలుగు", "മലയാളം", "ਪੰਜਾਬੀ", "ଓଡ଼ିଆ", "বাংলা"]

def process_image(image, language):
    # Placeholder for image processing logic
    return "Plant analysis results would appear here"

def process_question(message, history):
    # Placeholder for chat processing logic
    return f"Response to: {message}"

def text_to_speech(text):
    # Placeholder for TTS logic
    return None  # Return audio file path or audio data

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
            gr.Image("logo.png", elem_classes="logo")
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
                file_types=["image", "pdf"],
                file_count="single"
            )
            gr.Markdown("File type PDF or JPG, 250kb max file size")
            submit_btn = gr.Button("Submit", variant="primary")
    
    # Observations section with TTS
    with gr.Row():
        with gr.Column(elem_classes="textbox-container"):
            observations = gr.Textbox(
                label="Observations",
                placeholder="Analysis results will appear here...",
                interactive=False
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
    
    # Event handlers
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