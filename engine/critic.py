# engine/critic.py
import torch
from transformers import pipeline

# Load a lightweight DistilRoBERTa model fine-tuned for NLI (Natural Language Inference)
# This model determines if the 'hypothesis' is supported by the 'premise'.
device = 0 if torch.cuda.is_available() else -1
critic_pipe = pipeline("sentiment-analysis", 
                       model="cross-encoder/nli-distilroberta-base", 
                       device=device)

def verify_integrity(goal: str, output: str, lab_result: str = "") -> float:
    """
    The Superego: Evaluates if the output and lab results actually 
    satisfy the goal using objective NLI scoring.
    """
    # Premise: The goal + any empirical data from the lab
    premise = f"The goal is: {goal}. The lab verification result is: {lab_result}"
    
    # Hypothesis: What the agent actually claimed to do
    hypothesis = output[:500] # Limit context window for speed

    # NLI returns: 'entailment' (True), 'neutral', or 'contradiction' (False)
    # We want to see if the outcome ENTAILS the goal.
    result = critic_pipe([{"text": premise, "text_pair": hypothesis}])[0]
    
    label = result['label']
    score = result['score']

    # Map NLI labels to our 0.0 - 1.0 Hegelian scale
    if label == "entailment":
        return float(score)  # High score if logic follows
    elif label == "contradiction":
        return 0.1           # Punishment for failure
    else:
        return 0.4           # Neutral/Prose trap
