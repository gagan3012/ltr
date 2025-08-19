from ltr.dst import DistributionalSemanticsTracer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize DST
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tracer = DistributionalSemanticsTracer(model, tokenizer)

# Define pathways for analysis
prompt = "I saw an elephant in the forest. It had a large trunk."
correct_pathways = [["elephant", "trunk", "nose"]]
incorrect_pathways = [["forest", "trunk", "tree"]]

# Compute DSS
dss_score = tracer.compute_dss(
    prompt=prompt,
    correct_pathways=correct_pathways,
    incorrect_pathways=incorrect_pathways,
)

print(f"Distributional Semantics Strength: {dss_score:.4f}")
