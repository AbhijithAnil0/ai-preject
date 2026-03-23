from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def parse_query(user_query):

    query = user_query.lower()

    prompt = f"""
You convert user search queries into JSON filters.

Allowed keys:
object
color
event
action

Return JSON only.

User query: "{query}"
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # ---- Extract JSON ----
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        filters = {}
    else:
        json_text = text[start:end]

        try:
            filters = json.loads(json_text)
        except:
            filters = {}

    # -------------------------
    # Semantic correction layer
    # -------------------------

    # umbrella → person_U
    if "umbrella" in query:
        filters["object"] = "person_U"

    # people
    if "person" in query or "people" in query:
        filters["object"] = "person"

    # vehicles
    if "truck" in query:
        filters["object"] = "truck"

    if "car" in query:
        filters["object"] = "car"

    if "bus" in query:
        filters["object"] = "bus"

    # actions
    if "run" in query or "running" in query:
        filters["action"] = "run"

    if "walk" in query or "walking" in query:
        filters["action"] = "walk"

    if "stand" in query or "standing" in query:
        filters["action"] = "stand"

    return filters