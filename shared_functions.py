import json
import re
from typing import Any, Dict
from transformers import AutoTokenizer
import torch


def build_prompt(natural_text: str) -> str:
    instructions = (
        "Eres un extractor de órdenes de compra. Genera SOLO un JSON válido EXACTAMENTE con los campos requeridos.\n"
        "Reglas:\n"
        "- Usa null cuando un campo no exista.\n"
        "- \"buyer\" debe existir; si name/email/contact/addresses faltan, déjalos en null.\n"
        "- Si addresses está vacío o no existe -> \"addresses\": null.\n"
        "- Si purchases está vacío o no existe -> \"purchases\": null.\n"
        "- shipping es opcional; si falta -> \"shipping\": null.\n"
        "- Asegura que los tipos de datos principales sean correctos (quantity: entero; country uno de US/CA/GB/ES/CO/DE/FR).\n\n"
        "- Use null cuando no tengas información, y que NO inventes correos, teléfonos o códigos de descuento si no aparecen.\n"
        "- Respeta exactamente los nombres de los campos del esquema.\n"
        "- Estructura el problema paso a paso, razona por etapas.\n"
    )
    prompt = instructions + "Texto:\n" + natural_text + "\n\nRESPONDE SOLO CON EL JSON VÁLIDO (nada más):\nJSON:\n"
    return prompt

# Construcción del ejemplo de entrenamiento (prompt + target JSON)
def build_training_example(example: Dict[str, Any]) -> str:
    natural = example['natural_language']
    target_json = json.dumps(example['json_data'], ensure_ascii=False) # TODO: revisar indent=2
    prompt = build_prompt(natural)
    return prompt + target_json

# Tokenización (precompuesta) y construcción de datasets de tensores
def tokenize_example_textpair(textpair: str, max_length: int, tokenizer: AutoTokenizer, padding = 'max_length'):
    enc = tokenizer(textpair, truncation=True, max_length=max_length, padding= padding, return_tensors='pt') # todo: padding=False para probar, add_special_tokens=True
    labels = enc['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'labels': labels.squeeze(0)}


# Generar texto bruto desde el modelo - EN MODO EVALUACIÓN def generate_json_raw(text: str, max_new_tokens: int = 256) -> str:
def generate_json_raw(text: str, max_new_tokens: int, max_length: int, tokenizer: AutoTokenizer, model, device):
    prompt = build_prompt(text)
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # Heurística: recortar hasta la última '}' para obtener JSON válido en la mayoría de casos
    if "{" in decoded and "}" in decoded:
        last_brace = decoded.rfind("}")
        decoded = decoded[: last_brace + 1]
    return decoded    

def extract_json_from_text(text: str) -> Any:
    # Buscar primer bloque JSON {...}
    matches = re.findall(r"\{[\s\S]*\}", text)
    if matches:
        for m in matches:
            candidate = m.replace("'", '"')
            try:
                return json.loads(candidate)
            except Exception:
                continue
    # fallback: si hay 'JSON:' tomar lo que sigue
    if "JSON:" in text:
        part = text.split("JSON:")[-1].strip()
        part = part.strip("`\n ")
        try:
            return json.loads(part)
        except Exception:
            try:
                return json.loads(part.replace("'", '"'))
            except Exception:
                return None
    return None

