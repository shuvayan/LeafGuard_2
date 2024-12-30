import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_uploader as du
import os
from gtts import gTTS
from prediction_service.inference import predictor,make_prediction,DATASET_DIR

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
#UPLOAD_DIRECTORY = "uploads/"
UPLOAD_DIRECTORY = DATASET_DIR + "/Working_Images/crn.jpg"

os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Dash-Uploader for file uploads
du.configure_upload(app, UPLOAD_DIRECTORY)

# Languages supported
languages = ["English", "Hindi", "Kannada", "Marathi", "Telugu", "Malayalam", "Punjabi", "Odia", "Bengali"]

app.layout = html.Div([
    html.Div([
        html.Img(src="/assets/logo.png", style={"height": "50px", "float": "left", "margin-right": "10px"}),
        html.H1("Leaf Guard", style={"display": "inline-block", "margin": "0"}),
        html.H3("Intelligent Plant Disease Detection System", style={"margin-top": "5px"}),
    ], style={"text-align": "center", "margin-bottom": "20px"}),

    # Language Selection
    html.Div([
        html.Div([
            html.Button(lang, id=f"lang-{i}", n_clicks=0)
            for i, lang in enumerate(languages)
        ], style={"display": "flex", "justify-content": "center", "gap": "10px"}),
    ], style={"margin-bottom": "20px"}),

    # File Upload Section
    html.Div([
        du.Upload(
            id='uploader',
            text="Upload File",
            max_file_size=250,  # Max size in KB
            filetypes=["jpg", "jpeg", "png"],
        ),
        html.Div(id="file-uploaded", style={"margin-top": "10px", "text-align": "center"}),
        html.Button("Submit", id="submit-button", style={"margin-top": "20px", "display": "block", "margin": "10px auto"}),
    ], style={"text-align": "center", "margin-bottom": "20px"}),

    # Observations Section
    html.Div([
        html.H3("Observations", style={"margin-bottom": "10px"}),
        dcc.Textarea(id="observations", style={"width": "100%", "height": "150px"}, readOnly=True),
        html.Button(
            html.I(className="fa fa-microphone"), id="speak-observations", style={"margin-top": "10px"}
        ),
    ], style={"margin-bottom": "20px"}),

    # Question Input Section
    html.Div([
        html.H3("Type any questions here", style={"margin-bottom": "10px"}),
        dcc.Textarea(id="questions", style={"width": "100%", "height": "100px"}),
        html.Button(
            html.I(className="fa fa-microphone"), id="speak-questions", style={"margin-top": "10px"}
        ),
    ], style={"margin-bottom": "20px"})
])

# Callback to handle file upload and predictions
@app.callback(
    [Output("file-uploaded", "children"), Output("observations", "value")],
    [Input("uploader", "isCompleted"), Input("submit-button", "n_clicks")],
    [State("uploader", "fileNames")]
)
def process_file_upload(is_completed, n_clicks, filenames):
    if not is_completed or not filenames:
        return "No file uploaded.", ""

    # Assume single file upload
    filepath = os.path.join(UPLOAD_DIRECTORY, filenames[0])

    # Placeholder for prediction logic
    predictions = "Detected Leaf Disease: Rust. Suggested Treatment: Apply fungicide."  # Example result

    return f"Uploaded file: {filenames[0]}", predictions

# Callback to convert observations to speech
@app.callback(
    Output("speak-observations", "children"),
    Input("observations", "value")
)
def convert_to_speech(text):
    if not text:
        return ""

    tts = gTTS(text)
    speech_path = os.path.join(UPLOAD_DIRECTORY, "speech.mp3")
    tts.save(speech_path)

    return html.Audio(src=speech_path, controls=True)

if __name__ == "__main__":
    #app.run_server(debug=True)
    app.run_server(debug=True,host = '0.0.0.0',port = 8050)


