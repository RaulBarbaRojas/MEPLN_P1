"""
Módulo para entrenar y evaluar un modelo basado en HMM (CRF concretamente) para PoS tagging.

El modelo aquí empleado está originalmente descrito en https://colab.research.google.com/drive/1d7LO_0665DYw6DrVJXXautJAJzHHqYOm
Este script contiene múltiples modificaciones sobre dicho trabajo orientadas a mejorar la calidad de los resultados y a la\
investigación del funcionamiento del modelo en múltiples lenguajes.
"""

import random
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

from conllu import parse_incr
from sklearn_crfsuite import CRF, metrics, scorers


class ProcesadorTreeBank(ABC):
    """
    Clase para implementar un procesador de treebanks.
    """


    treebank : List[Tuple[List[str], List[str]]] = []


    @abstractmethod
    def procesar_treebank(self) -> None:
        """
        Método para procesar el treebank, obteniendo para cada palabra su PoS tag rellenando la variable `treebank`.
        """


class ProcesadorUDTreeBank(ProcesadorTreeBank):
    """
    Clase para implementar un procesador del treebank "Universal Dependencies".
    """

    def __init__(self, dataset_path : str | Path, annotated_file_relative_path : str | Path) -> None:
        """
        Constructor para la clase ProcesadorUDTreeBank.

        Args:
            dataset_path (str | Path): la ruta al conjunto de datos del Universal Dependencies TreeBank.
            corpus_relative_path (str | Path): la ruta relativa (a la anterior) donde se encuentra el fichero anotado con el que se entrenará.
        """
        super().__init__()
        self.dataset_path = dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)
        self.annotated_file_relative_path = annotated_file_relative_path if isinstance(annotated_file_relative_path, Path) else Path(annotated_file_relative_path)


    def procesar_treebank(self) -> None:
        """
        Método para procesar el Universal Dependencies TreeBank
        """

        with open(self.dataset_path / self.annotated_file_relative_path, encoding = 'utf-8', mode = 'r') as annotated_file:
            words, tags = [], []

            for ud_sentence_annotation in parse_incr(annotated_file):
                for word in ud_sentence_annotation:
                    words.append(word['form'])
                    tags.append(word['upostag'])

            self.treebank.append((words, tags))


class PreparacionDatos:
    """
    Clase para implementar lo necesario para preparar conjuntos de datos de cara a entrenamiento/evaluación de modelos CRF (HMM).
    """


    def __init__(self, treebank : List[Tuple[List[str], List[str]]]) -> None:
        """
        Constructor para la clase PreparacionDatos.

        Args:
            treebank (List[Tuple[List[str], List[str]]]): el treebank que conforma el conjunto de datos a emplear.
        """
        self.treebank = treebank


    @abstractmethod
    def _extraer_caracteristicas_de_palabra(self, oracion : str, indice_palabra : int) -> Dict[str, Any]:
        """
        Método para extraer características de un conjunto de datos dado.

        Args:
            oracion (str): la oración a la que pertenece la palabra cuyas características se van a extraer.
            indice_palabra (int): el índice que tiene la palabra en la oración.

        Returns:
            El diccionario de características con las que entrenar el modelo. 
        """


    def _procesar_treebank(self, treebank : List[Tuple[List[str], List[str]]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Método para procesar un treebank generando un conjunto de datos hábil para entrenamiento/evaluación.

        Args:
            treebank (List[Tuple[List[str], List[str]]]): el treebank (o subconjunto del mismo) que se quiere procesar.

        Returns:
            Una tupla (X, y) con datos hábiles para entrenamiento y/o evaluación de modelos.
        """
        X, y = [], []

        for oracion, etiquetas in treebank:
            caracteristicas_oracion, etiquetas_oracion = [], []

            for indice_palabra in range(len(oracion)):
                caracteristicas_oracion.append(self._extraer_caracteristicas_de_palabra(oracion, indice_palabra))
                etiquetas_oracion.append(etiquetas[indice_palabra])

            X.append(caracteristicas_oracion)
            y.append(etiquetas_oracion)

        return X, y


    def preparar_datos(self, pct_entrenamiento : float = 0.8) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
        """
        Método para preparar los datos de cara al entrenamiento/evaluación del modelo.

        Args:
            pct_entrenamiento (float): el porcentaje de datos que se van a emplear en el entrenamiento del modelo.

        Returns:
            Una tupla (X_train, y_train, X_test, y_test) con los datos particionados para realizar entrenamiento y evaluación del modelo.
        """
        random.shuffle(self.treebank)
        treebank_entrenamiento = self.treebank[:round(len(self.treebank) * pct_entrenamiento)]
        treebank_evaluacion = self.treebank[round(len(self.treebank) * pct_entrenamiento):]

        return *self._procesar_treebank(treebank_entrenamiento), *self._procesar_treebank(treebank_evaluacion)


class PreparacionDatosAlfabetoLatino(PreparacionDatos):
    """
    Clase para implementar lo necesario para preparar datos en una lengua con alfabeto latino.
    """


    def _extraer_caracteristicas_de_palabra(self, oracion : str, indice_palabra : int) -> Dict[str, Any]:
        """
        Método para extraer características de un conjunto de datos dado.

        Args:
            oracion (str): la oración a la que pertenece la palabra cuyas características se van a extraer.
            indice_palabra (int): el índice que tiene la palabra en la oración.

        Returns:
            El diccionario de características con las que entrenar el modelo. 
        """
        return {
            'word':oracion[indice_palabra],
            'is_first':indice_palabra==0,
            'is_last':indice_palabra ==len(oracion)-1,
            'is_capitalized':oracion[indice_palabra][0].upper() == oracion[indice_palabra][0],
            'is_all_caps': oracion[indice_palabra].upper() == oracion[indice_palabra],
            'is_all_lower': oracion[indice_palabra].lower() == oracion[indice_palabra],
            'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',oracion[indice_palabra])))),
            'prefix-1':oracion[indice_palabra][0],
            'prefix-2':oracion[indice_palabra][:2],
            'prefix-3':oracion[indice_palabra][:3],
            'prefix-3':oracion[indice_palabra][:4],
            'suffix-1':oracion[indice_palabra][-1],
            'suffix-2':oracion[indice_palabra][-2:],
            'suffix-3':oracion[indice_palabra][-3:],
            'suffix-3':oracion[indice_palabra][-4:],
            'prev_word':'' if indice_palabra == 0 else oracion[indice_palabra-1],
            'next_word':'' if indice_palabra < len(oracion) else oracion[indice_palabra+1],
            'has_hyphen': '-' in oracion[indice_palabra],
            'is_numeric': oracion[indice_palabra].isdigit(),
            'capitals_inside': oracion[indice_palabra][1:].lower() != oracion[indice_palabra][1:]
    }


if __name__ == '__main__':

    treebanks_por_lenguaje = {
        'Inglés' : 'UD_English-GUM/en_gum-ud-train.conllu',
    }

    for idioma, ruta_treebank in treebanks_por_lenguaje.items():
        # Paso 1: Creamos el procesador del Treebank de Universal Dependencies
        procesador_ud = ProcesadorUDTreeBank(r'C:\Users\raulb\Desktop\Master\ENLP\datasets\ud-treebanks-v2.5', ruta_treebank)
        procesador_ud.procesar_treebank()

        # Paso 2: Creamos conjunto de datos de entrenamiento/evaluación
        match idioma:
            case 'Inglés' | 'Español':
                generador_datos = PreparacionDatosAlfabetoLatino(procesador_ud.treebank)
            case _:
                raise RuntimeError('Idioma no reconocido por el script de entrenamiento de PoS taggers CRF')

        X_train, y_train, X_test, y_test = generador_datos.preparar_datos()

        # Paso 3: Creamos el modelo CRF (derivado de HMM). El crédito de este modelo es del autor del notebook mostrado al inicio de este script
        modelo = CRF(algorithm = 'lbfgs', c1 = 0.01, c2 = 0.1, max_iterations = 100, all_possible_transitions = True)

        tiempo_inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        tiempo_fin = time.perf_counter()
        print(f'Tiempo de entrenamiento [{idioma, ruta_treebank}]: ', round(tiempo_fin - tiempo_inicio, 4))

        # Paso 4: Evaluamos el modelo
