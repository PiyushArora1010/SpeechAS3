import gradio as gr
import numpy as np
import torch
import argparse

import warnings
warnings.filterwarnings("ignore")

from module.model import load_model

argparser = argparse.ArgumentParser()
argparser.add_argument("--path", type=str, default="model.pth")
args = argparser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
        model = load_model(DEVICE, args.path).eval()

        def verify_audio(audio1):
            audio1 = audio1[1] / np.max(np.abs(audio1[1]))
            audio1 = torch.tensor(audio1, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            sim = model(audio1)
            return str(sim.cpu().detach().numpy()[0])
        
        audio1 = gr.components.Audio(source="upload", label="Upload your first audio file", type="numpy")
        outputs = gr.outputs.Label(label="Spoof (0) or Real (1)")
        gr.Interface(fn=verify_audio, inputs=[audio1], outputs=outputs).launch(share=True)