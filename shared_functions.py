import json
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
import math

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

# Construcción del prompt
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
    enc = tokenizer(textpair, truncation=True, max_length=max_length, padding=padding, return_tensors='pt', add_special_tokens=True)
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


# ---------------------------------------------------- FUNCIONES DE VALIDACIÓN y EVALUACIÓN ------------------------------------------------------------------------------------------------
## Funciones para procesar datos por lotes y extraer JSON de texto generado por el modelo
def generate_json_raw_batch( texts: List[str], tokenizer, model, device, max_new_tokens: int, max_length: int, batch_size: int = 8):
    outputs = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    #eos_brace_id = tokenizer.encode("}", add_special_tokens=False)[0]

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating", total=math.ceil(len(texts)/batch_size)):
        batch = texts[i:i + batch_size]
        prompts = [build_prompt(t) for t in batch]

        enc = tokenizer( prompts, return_tensors='pt', truncation=True, padding="longest", max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.inference_mode():
            model.eval()
            out = model.generate( 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=max_new_tokens, 
                do_sample=False, 
                pad_token_id=pad_id, 
                eos_token_id=tokenizer.eos_token_id, 
                use_cache=True
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True) # Decodificar outputs en lote

        # Recorte final
        cleaned = []
        for d in decoded:
            d = (d.replace("“", '"').replace("”", '"').replace("’", "'"))

            if "{" in d and "}" in d:
                first = d.find("{")
                last = d.rfind("}")
                d = d[first:last+1]
            cleaned.append(d)

        outputs.extend(cleaned)

    return outputs


def extract_json_from_text(text: str):
    """
    Escanea llaves para encontrar un bloque JSON bien balanceado.
    """
    marker = '{"buyer":'
    pos = text.find(marker)
    if pos != -1:
        start = text.find(marker)
    else: # pos == -1
        marker = "\nJSON:\n"
        pos = text.find(marker)
        if pos == -1:
            start = text.find("{")
        else:
            start = text.find("{", pos + len(marker)) # Buscar la primera llave '{' después del marcador
            if start == -1:
                start = text.find("{")

    if start == -1:
        return None

    brace_count = 0
    in_json = False

    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
            in_json = True
        elif text[i] == "}":
            brace_count -= 1

            # Si brace_count llega a 0 => JSON completo
            if in_json and brace_count == 0:
                candidate = text[start:i+1]

                # intentar parsear
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # intento reemplazando comillas simples
                    try:
                        return json.loads(candidate.replace("'", '"'))
                    except Exception:
                        return None

    return None