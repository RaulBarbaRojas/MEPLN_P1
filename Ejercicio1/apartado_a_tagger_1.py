"""
Módulo para resolver el apartado a) del primer ejercicio de la práctica I.
"""

import time

from transformers import pipeline


if __name__ == '__main__':

    # Utilización de un tagger en una lengua romance (español)
    etiquetador_espanol = pipeline('token-classification', model='PlanTL-GOB-ES/roberta-large-bne-capitel-pos')
    with open('assets/INPUT_RAW_espanol.txt', mode = 'r', encoding = 'utf-8') as archivo_entrada_espanol:
        entrada_espanol = archivo_entrada_espanol.read()

    for idx_page in range(2, 41):
        entrada_espanol = entrada_espanol.replace(f'\n{idx_page}\n', '\n')

    entrada_espanol = entrada_espanol.replace('\n', ' ').replace('-', ',').replace('• ', '')
    entrada_espanol = entrada_espanol[2:]

    with open('assets/INPUT_RAW_espanol_processed.txt', mode = 'w') as archivo_entrada_procesada_espanol:
        archivo_entrada_procesada_espanol.write(entrada_espanol)

    start_time = time.perf_counter()
    etiquetas_espanol = etiquetador_espanol(entrada_espanol)
    end_time = time.perf_counter()
    with open('out/OUTPUT_RAW_espanol.txt', mode = 'w', encoding='utf-8') as archivo_salida_espanol:
        archivo_salida_espanol.write(str(etiquetas_espanol))

    print('Etiquetador español: ', end_time - start_time)
