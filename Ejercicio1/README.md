# Ejercicio 1

El primer ejercicio de la práctica dispone de dos apartados principales, que quedan descritos en las siguientes secciones de este mismo documento.

## Apartado a

Este primer apartado del ejercicio 1 de la práctica tiene como objetivo buscar y descargar etiquetadores de uso libre que estén pre-entrenados para dos idiomas: (1) inglés y (2) alguna lengua romance (como el español, que es el que se emplea en la resolución). Se 

### Etiquetador 1

El primer etiquetador emplea el modelo BERT y se encuentra accesible en [este repositorio de HuggingFace](https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos) con licencia Apache 2.0, una licencia escrita por Apache Software Foundation que permite al usuario del software *la libertad para usarlo para cualquier propósito, distribuirlo, modificarlo y distribuir versiones modificadas de ese software*, aunque no ofrece garantías sobre el uso del software tal y como se expone en los [términos y condiciones de la licencia Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.html). Este etiquetador ha sido desarrollado por el Plan de Impulso de las Tecnologías de Lenguaje del Gobierno de España, cuyo objetivo es propulsar la industria del procesamiento del lenguaje natural, la traducción automática y los sistemas conversacionales en España, tal y como se describe en su [página web oficial](https://plantl.mineco.gob.es/Paginas/index.aspx).

Sobre el etiquetador empleado, se trata de un etiquetador BERT, un modelo que se basa en la arquitectura de transformadores y que concretamente emplea un transformer bidireccional para tratar de maximizar su rendimiento, principalmente en tareas que se basan en datos secuenciales como es el caso de las tareas de procesamiento de lenguaje natural. El modelo fue originalmente propuesto en 2019 (KENTON, Jacob Devlin Ming-Wei Chang; TOUTANOVA, Lee Kristina. Bert: Pre-training of deep bidirectional transformers for language understanding. En Proceedings of naacL-HLT. 2019. p. 2.), y desde entonces se ha utilizado ampliamente para resolver múltiples tareas de aprendizaje automático en general y procesamiento de lenguaje natural en particular, como es el caso del problema de PoS tagging que se resuelve en este apartado. Por tanto, nos encontramos ante un modelo basado en redes neuronales con el que resolver esta tarea de PoS tagging en un idioma romance como es el español.

Sobre el procesamiento de la entrada, se realizaron múltiples pruebas sobre el etiquetador descargado. Tras las pruebas realizadas, ciertos preprocesamientos en la entrada dieron lugar a salidas más coherentes (principalmente por el tokenizador que utiliza internamente el modelo y no tanto por la clasificación de palabras en sí). De este modo, se aplicaron los siguientes preprocesamientos básicos:

- Se leyó la entrada con el formato UTF-8, lo que permitió codificar tildes (entre otros) aspectos típicos en el español.

- Sobre el texto en UTF-8, se eliminaron de forma automática los números de página del discurso (a través del patrón: salto de línea - número de página - salto de línea).

- Se reemplazó el carácter "-" empleado en el discurso para hacer aclaraciones por comas ",".

- Se eliminaron los guiones redondos que formaban listas no numeradas en el discurso.

Con respecto a la salida, observamos que el modelo, basado en redes neuronales, devuelve una categoría (e.g., NOUN) y una puntuación (e.g. 0.9815062), que se corresponde con la probabilidad que el etiquetador asocia la categoría que propone a cada token (palabra en este caso) de la secuencia de entrada. Además de lo anterior, observamos más información en la salida, concretamente observamos que para cada token de la entrada, se tiene lo siguiente:

- Un atributo "index" que muestra la posición del token en la cadena de entrada (1 para la primera frase, 2 para la segunda, y así sucesivamente).

- Un atributo "start" y un atributo "end" que se refieren a los índices en que empieza y termina el token analizado (esto es, el token empieza en el índice que dice "start" de forma inclusiva, y termina en el que dice "end" de forma no inclusiva).

- Un atributo "word" que muestra el token de la cadena de entrada. Dicho token empieza por un símbolo "extraño" (Ġ) en la mayoría de los casos. Esto se debe a que es un carácter que emplea el tokenizador interno del modelo para separar los tokens. Así cada palabra empieza por dicho símbolo excepto (si lo hubiera) el símbolo de puntuación final.

El modelo se ejecutó sobre un texto de 11379 palabras correspondiente al discurso de investidura de una de las caras políticas de la España de nuestra época y que se encuentra accesible en el siguiente [enlace](https://www.pp.es/sites/default/files/documentos/23.09.26_discurso_investidura_anf.pdf). La razón por la que se utilizó este texto y no otro se debe a la estrategia seguida para encontrar un texto de dicho tamaño. Primeramente, se estimó que un texto de 10000 palabras equivaldría a unas 30 o 40 páginas en función del tamaño de letra. A continuación, se procedió a analizar qué tipo de documentos podrían tener esta longitud y que no difieran excesivamente del corpus sobre el que se entrenó el modelo (que es un corpus de noticias generado por el Plan de Impulso de Tecnologías de Lenguaje Humano). El resultado del análisis fue que el tipo de texto más adecuado para ejecutar el modelo sobre él sería un discurso, por lo que se realizó una búsqueda en un motor de búsqueda en Internet filtrando por textos en formato PDF para encontrar un discurso del tamaño deseado. Dicha búsqueda resultó en el documento que se emplea para este apartado.

Observando la salida, podemos ver cómo el modelo funciona bien, a excepción de algunos fallos propios del conjunto de entrenamiento sobre el que se entrenó el modelo (véase como la palabra "investidura" está mal tokenizada por el tokenizador que emplea internamente, lo que lleva al modelo a asociar dos categorías, una a "investi" y otra a "dura", probablemente porque el adjetivo "dura" es frecuente en el conjunto de entrenamiento que, como ya sabemos, se trata de noticias extraídas por el Plan de Impulso de Tecnologías de Lenguaje Humano del Gobierno de España). A continuación se muestra un breve extracto de la salida del modelo para la parte de la frase "Alcalde de Madrid", donde las tres palabras se tokenizan y clasifican correctamente de forma automática por el modelo.

```python
[
    {
        'entity': 'NOUN',
        'score': np.float32(0.9815062),
        'index': 56, 'word': 'ĠAlcalde',
        'start': 283, 'end': 290
    },
    {
        'entity': 'ADP',
        'score': np.float32(0.9970836),
        'index': 57,
        'word': 'Ġde',
        'start': 291,
        'end': 293
    },
    {
        'entity': 'PROPN',
        'score': np.float32(0.99465406),
        'index': 58,
        'word': 'ĠMadrid',
        'start': 294,
        'end': 300
    },
...
]
```

En términos de rendimiento, el modelo tardó aproximadamente 1.1352 segundos en todas las palabras de la entrada, por lo que podemos decir que tuvo un gran rendimiento (en torno a 10000 palabras por segundo).


### Etiquetador 2

El segundo de los etiquetadores que se proponen para la resolución de la práctica es un modelo de la arquitectura BERT (por tanto, un modelo que emplea el mecanismo de atención para tratar de conseguir mejores resultados centrándose en aspectos más concretos, en este caso, de las frases de entrada). El modelo pertenece al Instituto de Investigación en Computación de Catar (QCRI, cuya página web es accesible en [este enlace](https://www.hbku.edu.qa/en/qcri)), concretamente al departamento de Tecnologías de Lenguas Arábicas que, aunque se centran en lenguas arábicas, no dejan de lado el inglés y han creado un modelo para realizar PoS tagging en inglés.

Dado que la arquitectura del modelo se basa nuevamente en la arquitectura BERT, las descripciones proporcionadas sobre dicha arquitectura en el etiquetador anterior son igualmente aplicables al presente etiquetador. Dado que este modelo está entrenado con un conjunto de datos multilingüe, podría potencialmente ser utilizado para el español. Sin embargo, dado que ya se encontró un modelo entrenado específicamente sobre corpus en español, se utilizó este último para la etiquetación sobre una lengua romance (el español concretamente, tal y como se ha descrito en anteriormente), mientras que este modelo quedó relevado a la etiquetación de la entrada en inglés.

En lo que respecta al texto de entrada, el texto se puede encontrar originalmente en este [repositorio de Github](https://github.com/first20hours/google-10000-english/blob/master/google-10000-english-no-swears.txt). Se trata de un texto que puede utilizarse libremente para motivos de investigación, personales y educativos, tal y como se especifica en la licencia LDC que los autores originales (Google, pues este texto fue obtenido por el creador del repositorio a partir del corpus del trillón de palabras web de Google, y el autor respetó en todo momento la licencia original, tal y como se observa en el apartado "License" del repositorio). Dicha entrada únicamente recibió el preprocesamiento de sustituir los saltos de línea por espacios en blanco con el objetivo de facilitar aún más la labor del tokenizador que lleva incluido el modelo del que se dispone. Nuevamente nos encontramos ante un modelo que no requiere ningún tipo de preprocesamiento de la entrada para su funcionamiento, pues internamente descarga otros modelos que componen un flujo más grande y que se encargan de realizar el procesamiento necesario para resolver la tarea de etiquetación de la entrada proporcionada.

Considerando la salida mostrada por el etiquetador, observamos que se mantiene el mismo patrón que el modelo anterior, pues ambos modelos proceden de un mismo repositorio de modelos: [HuggingFace](https://huggingface.co/), un repositorio de modelos y conjunto de datos de aprendizaje automático que ha adquirido gran relevancia últimamente por disponer de modelos que resuelven muchas tareas actuales (sobre todo de procesamiento de lenguaje natural), y hacerlo en diferentes tecnologías y marcos de desarrollo (como podrían ser PyTorch, TensorFlow, o incluso modelos de aprendizaje automático implementados en otros diferentes). Tal y como se muestra a continuación, la salida es idéntica al caso anterior, donde el resultado es un fichero en formato JSON (lista JSON de objetos JSON concretamente), donde cada palabra dispone de información relacionada con su comienzo, fin, categoría  y certeza sobre dicha categoría (la probabilidad de que lo sea). Observando el resultado, observamos como el modelo ha funcionado mejor que el modelo para el español anterior, lo que es un ejemplo de que una misma arquitectura puede conseguir mejores o peores resultados dependiendo de los datos sobre los que entrena (es mucho más fácil aprender para el inglés y más difícil para el español, como se ha visto en la teoría de esta asignatura).

```python
[
    {
        'entity': 'DT',
        'score': np.float32(0.99597687),
        'index': 1,
        'word': 'the',
        'start': 0,
        'end': 3
    },
    {
        'entity': 'IN',
        'score': np.float32(0.9989398),
        'index': 2,
        'word': 'of',
        'start': 4,
        'end': 6
    },
    {
        'entity': 'CC',
        'score': np.float32(0.9993349),
        'index': 3,
        'word': 'and',
        'start': 7,
        'end': 10
    },
...
]
```




## Apartado b

ToDo
