import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import base64
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-music1",
            children=html.Div(["Drag and drop or click to select a file"]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="file-name-output1"),
        html.Div(id="spectrogram-container1"),
        dcc.Upload(
            id="upload-music",
            children=html.Div(["Drag and drop or click to select a file"]),
            style={
                "width": "50%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="file-name-output"),
        html.Audio(id="audio-player", controls=True),
        html.Div(id="spectrogram-container"),
    ]
)

@app.callback(
    Output("file-name-output1", "children"),
    Output("spectrogram-container1", "children"),
    Input("upload-music1", "contents"),
    State("upload-music1", "filename"),
)
def update_spectrogram1(contents, filename):
    if contents is not None:
        # Split the contents using the comma delimiter
        content_type, content_string = contents.split(",")

        # Decode the Base64 encoded string
        audio_bytes = base64.b64decode(content_string)

        # Load the audio data and sampling rate using librosa
        audio_data, sampling_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # Generate the spectrogram using librosa
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sampling_rate, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram 1")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Convert the matplotlib figure to an image
        spectrogram_image = io.BytesIO()
        plt.savefig(spectrogram_image, format="png")
        plt.close()

        # Create a data URI for the spectrogram image
        spectrogram_src = f"data:image/png;base64,{base64.b64encode(spectrogram_image.getvalue()).decode()}"

        return f"Selected file 1: {filename}", html.Img(src=spectrogram_src)

    return None, None


@app.callback(
    Output("audio-player", "src"),
    Output("file-name-output", "children"),
    Output("spectrogram-container", "children"),
    Input("upload-music", "contents"),
    State("upload-music", "filename"),
)
def update_audio_player(contents, filename):
    if contents is not None:
        # Read the contents of the uploaded file as bytes
        content_type, content_string = contents.split(",")
        audio_bytes = base64.b64decode(content_string)
        audio_data, sampling_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # Generate the spectrogram using librosa
        spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=sampling_rate, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Convert the matplotlib figure to an image
        spectrogram_image = io.BytesIO()
        plt.savefig(spectrogram_image, format="png")
        plt.close()

        # Create a data URI for the spectrogram image
        spectrogram_src = f"data:image/png;base64,{base64.b64encode(spectrogram_image.getvalue()).decode()}"

        # Create a data URI for the audio file
        audio_src = f"data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}"

        return audio_src, f"Selected file: {filename}",  html.Img(src=spectrogram_src)

    return None, None, None


if __name__ == "__main__":
    app.run_server(debug=True)
