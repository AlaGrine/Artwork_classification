
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

##############################################
# 1. Setup class names
##############################################
class_names = ['art_nouveau',
 'baroque',
 'expressionism',
 'impressionism',
 'post_impressionism',
 'realism',
 'renaissance',
 'romanticism',
 'surrealism',
 'ukiyo_e']

##############################################
# 2. Model and transforms preparation 
##############################################

# 2.1 Create EfficientNet_B2 model
EfficientNetB2_model, EfficientNetB2_transforms = create_effnetb2_model(num_classes=10,is_TrivialAugmentWide=False)

# 2.2 Load saved weights (from our trained PyTorch model)
EfficientNetB2_model.load_state_dict(
    torch.load(
        f="EfficientNet_B2_FT.pth",
        map_location=torch.device("cpu"),  # load to CPU because we will use the free HuggingFace Space CPUs.
    )
)

##############################################
# 3. Create prediction function
##############################################
def prediction(img) -> Tuple[Dict, float]:
    """returns prediction probabilities and prediction time.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = EfficientNetB2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    EfficientNetB2_model.eval()
    with torch.inference_mode():
        # Get prediction probabilities
        pred_probs = torch.softmax(EfficientNetB2_model(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class.
    # This is the required format for Gradio's output parameter.
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time
    
##############################################
# 4. Gradio app
##############################################
# 4.1 Create title, description and article strings
title = "Artwork Classification 🎨"
description = "An EfficientNetB2 computer vision model to classify artworks."
article = "Created with PyTorch."

# 4.2 Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# 4.3 Create the Gradio demo
demo = gr.Interface(fn=prediction, # mapping function from input to output
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # 1st output: pred_probs
                             gr.Number(label="Prediction time (s)")], # 2nd output: pred_time
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# 4.4 Launch the Gradio demo!
demo.launch()
