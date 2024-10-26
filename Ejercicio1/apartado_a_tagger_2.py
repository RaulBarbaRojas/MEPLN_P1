"""
Módulo para resolver el apartado a) del primer ejercicio de la práctica I (con un segundo tagger).
"""

import time

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline


if __name__ == '__main__':

    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

    with open('assets/apartado_a/INPUT_RAW_ingles.txt', mode = 'r', encoding = 'utf-8') as archivo_entrada_ingles:
        entrada_ingles = archivo_entrada_ingles.read()

    entrada_ingles = entrada_ingles.replace('\n', ' ')

    with open('assets/apartado_a/INPUT_RAW_ingles_processed.txt', mode = 'w') as archivo_entrada_procesada_ingles:
        archivo_entrada_procesada_ingles.write(entrada_ingles)

    start_time = time.perf_counter()
    etiquetas_ingles = pipeline(entrada_ingles)
    end_time = time.perf_counter()
    with open('out/apartado_a/OUTPUT_RAW_ingles.txt', mode = 'w', encoding='utf-8') as archivo_salida_ingles:
        archivo_salida_ingles.write(str(etiquetas_ingles))

    print('Etiquetador inglés: ', end_time - start_time)
