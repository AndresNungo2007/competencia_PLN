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
    prompt = instructions + "Texto:\n" + natural_text + "\n\nRESPONDE SOLO CON EL JSON VÁLIDO (nada más):\nJSON:\n"
    return prompt
