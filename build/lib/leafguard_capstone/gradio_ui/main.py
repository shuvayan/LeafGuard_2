# leafguard_capstone/gradio_ui/main.py
from leafguard_capstone.gradio_ui.gradio_app import demo

def launch_ui():
    demo.launch()

if __name__ == "__main__":
    launch_ui()