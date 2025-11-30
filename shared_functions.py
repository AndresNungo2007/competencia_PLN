import json
from typing import Any, Dict
import numpy as np

# --------------------------------------------------- FUNCIONES COMPARTIDAS ENTRE ENTRENAMIENTO, VALIDATION Y EVALUATION ----------------------------------------------------------
def print_time_execution(description, start, end):
    str_log = ""
    duration = end - start
    if duration > 3600:
        horas = duration // 3600
        minutos = (duration % 3600) // 60
        segundos = duration % 60
        str_log = f"Tiempo {description}: {horas:.0f} horas, {minutos:.0f} minutos y {segundos:.2f} segundos"
    elif duration > 60:
        minutos = duration // 60
        segundos = duration % 60
        str_log = f"Tiempo {description}: {minutos:.0f} minutos y {segundos:.2f} segundos"
    else:
        str_log = f"Tiempo {description}: {duration:.6f} segundos"
    return str_log

# Construcción del prompt. Del score más alto.
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
    prompt = instructions + "Texto:\n" + natural_text + "\n\nJSON:\n"
    return prompt


# --------------------------------------------------- FUNCIONES DE ENTRENAMIENTO ------------------------------------------------------------------------------------------------
# Tokenización (precompuesta) y construcción de datasets de tensores. Para entrenamiento padding debe ser False
def tokenize_example_textpair(textpair, max_length, tokenizer, padding=False):
    enc = tokenizer(textpair, truncation=True, max_length=max_length, padding=padding, return_tensors='pt', add_special_tokens=True) # este estaba en false: add_special_tokens
    labels = enc['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        'input_ids': enc['input_ids'].squeeze(0),
        'attention_mask': enc['attention_mask'].squeeze(0),
        'labels': labels.squeeze(0)
    }

# Construcción del ejemplo de entrenamiento (prompt + target JSON)
def build_training_example(example: Dict[str, Any]) -> str:
    natural = example['natural_language']
    target_json = json.dumps(example['json_data'], ensure_ascii=False)
    prompt = build_prompt(natural)
    return prompt + target_json

# Esto para detectar MAX_LENGTH y GEN_MAX_NEW_TOKENS en datos de entrenamiento
def medir_longitudes_tokens(dataset, tokenizer, max_ejemplos=None):
    prompt_lens = []
    json_lens = []
    full_lens = []
    
    for i, ex in enumerate(dataset):
        if max_ejemplos is not None and i >= max_ejemplos:
            break
        
        # Texto natural y JSON objetivo de tu dataset
        natural = ex["natural_language"]
        target_json_str = json.dumps(ex["json_data"], ensure_ascii=False)
        
        # Construir el mismo prompt que usas en train/eval
        prompt = build_prompt(natural)
        full_text = prompt + target_json_str
        
        # Tokenizar SIN truncar ni hacer padding fijo
        enc_prompt = tokenizer(
            prompt,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        enc_full = tokenizer(
            full_text,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        
        lp = len(enc_prompt["input_ids"])
        lf = len(enc_full["input_ids"])
        lj = lf - lp  # aproximación a longitud del JSON
        
        prompt_lens.append(lp)
        full_lens.append(lf)
        json_lens.append(lj)
    
    stats = {
        "prompt_mean": float(np.mean(prompt_lens)),
        "prompt_p95": float(np.percentile(prompt_lens, 95)),
        "prompt_p99": float(np.percentile(prompt_lens, 99)),
        
        "json_mean": float(np.mean(json_lens)),
        "json_p95": float(np.percentile(json_lens, 95)),
        "json_p99": float(np.percentile(json_lens, 99)),
        
        "full_mean": float(np.mean(full_lens)),
        "full_p95": float(np.percentile(full_lens, 95)),
        "full_p99": float(np.percentile(full_lens, 99)),
        "full_max": int(np.max(full_lens)),
    }
    return stats