<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Iconos de sobre de correo electrónico</title>
    <!-- Incluye la biblioteca Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
</head>
<body>

# <h1 align=center> **`PROYECTO INDIVIDUAL Nº1`**</h1> <!-- omit in toc --> 

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1> <!-- omit in toc --> 

# <h3 align=center>**` Diciembre de 2023 `**</h2> <!-- omit in toc --> 

# <h2 align=center>**` Miguel Angel Dallanegra Vilches - DATA Part Time 05 `** </h2> <!-- omit in toc --> 

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>
<hr>

  
- [Información General](#información-general)
- [Desarrollo](#desarrollo)
- [ETL](#etl)
- [NLP](#nlp)
- [EDA](#eda)
- [Tablas y Funciones para FastApi y modelo de Recomendaciones](#tablas-y-funciones-para-fastapi-y-modelo-de-recomendaciones)
  - [Tablas](#tablas)
  - [Funciones](#funciones)
    - [Desarrollo API](#desarrollo-api)
    - [Modelo de Recomendación](#modelo-de-recomendación)
- [Deploy FastAPI \& Render](#deploy-fastapi--render)
- [Video](#video)
- [Conclusiones](#conclusiones)
- [Contacto](#contacto)

## Información General

Se tiene una base de datos compuesta de tres archivos comprimidos que contienen informacion acerca de juegos y programas de computadora, sin procesar (en formato RAW), que deben ser trabajados para poder interpretarlos y tomar informacion para luego poder hacer modelos de recomendacion y otros pedidos "estadisticos".

## Desarrollo

El proceso de trabajo consistirá en 4 partes:

- Se deberan extraer y transformar los datos que llegan sin procesar y con errores, los cuales se encuentran en la carpeta [_data](https://github.com/mdallanegra/LAB_PI_01/tree/main/_data).
- Se hará un análisis de sentimiento y análisis exploratorio para entender la informacion que contiene la base de datos.
- Se deberá crear un modelo de recomendación mediante ML, para el cual se crearan tablas especificas y de menor tamaño para mejor usabilidad.
- Se hará el deploy de los modelos de recomendacion en FastAPI y luego en render. Los archivos correspondientes a esta etapa se encontrarán en la carpeta principal de github [(LAB_PI_01)](https://github.com/mdallanegra/LAB_PI_01) para el caso de [requirements.txt](https://github.com/mdallanegra/LAB_PI_01/blob/main/requirements.txt) y [FastAPI.py](https://github.com/mdallanegra/LAB_PI_01/blob/main/FastAPI.py). Las tablas optimizadas para los modelos se encontraran en  [FastAPI](https://github.com/mdallanegra/LAB_PI_01/tree/main/FastAPI).

Las tres primeras partes se haran paso por paso en archivos de jupyter notebook, generando nuevas tablas y para el deploy se creará un archivo python. Tambien es importante aclarar que todos los archivos poseen aclaraciones para cada parte de los procesos, explicándolos y emitiendo algunas conclusiones.

## ETL

El proceso de extracción y transformación de los datos consiste en 4 archivos notebook que se encuentran en [_notebooks](https://github.com/mdallanegra/LAB_PI_01/tree/main/_notebooks)
- El primero es el de descompresion de las bases de datos y la extraccion de la información. Este es el archivo: [Lab_PI_01.1.Extraccion.ipynb](
https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.1.Extraccion.ipynb)
- Los tres siguientes consisten en la transformacion y limpieza de cada una de las tablas por separado haciendo mas facil de manejar las etapas del procesamiento de las mismas. Estas son:
  -  Archivo Steam Games, que contiene informacion de cada uno de los programas en la base de datos de la empresa Steam.
[Lab_PI_01.2.Transformacion.01.steam_games.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.2.Transformacion.01.steam_games.ipynb)
- Archivo User Reviews, que consta de las revisiones o críticas de los usuarios de cada juego, dando su opinión y emitiendo su voto referente a si recomiendan el producto o no. [Lab_PI_01.2.Transformacion.02.user_reviews.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.2.Transformacion.02.user_reviews.ipynb)
- Archivo Users Items, que consta de las estadísticas de juego o tiempo de trabajo en cada uno de los productos generados por cada usuario. [Lab_PI_01.2.Transformacion.03.users_items.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.2.Transformacion.03.users_items.ipynb)

Todos estos archivos han sido limpiados, eliminando columnas innecesarias, borrando elementos duplicados o vacíos, reordenando sus datos y finalmente han sido almacenados en formato parquet para optimizar el trabajo posterior pensando tambien que el espacio que ocupan es notablemente menor al de otros formatos, lo que será de gran utilidad para el deploy y el guardado de las mismas en GitHub.

## NLP

En esta parte se genra un notebook con el Procesamiento de Lenguaje Natural, para comprender las reviews obtenidas de lo escrito por los usuarios y generando finalmente una valoración basada en la siguiente escala: '0' si es una review mala, '1' si es neutral y '2' si es positiva.

Antes de realizar el análisis de sentimiento, se tuvo que tener en cuenta las stopwords y palabras utiles del lenguaje referido a juegos y programas, y ademas se debió lematizar y tokenizar las reviews para que el modelo pueda hacer el trabajo necesario para replicar la escala pedida. 

Para el procesamiento de texto se utilizó vader-lexicon del modulo nltk que asigna valores a las palabras y luego pondera el valor total para asignarle una nota. Es importante aclarar que vader-lexicon no es tan exacto al analizar lenguaje coloquial o ciertas formas idiomaticas, por lo que puede no entender dobles sentidos, sarcasmos o ironías. 

Además si las palabras no están en el diccionario, puede asignarle en forma incorrecta un valor. De todas formas es el que mejor implementó los resultados frente a TextBlob o Transformers. En el caso de Transformers basado en el modelo distilbert-base-uncased debería haber tenido mejor rendimiento, pero probablemente fue ejecutado incorrectamente. 

Tambien se probó StanfordNLP, pero no pudo ser puesto en marcha, aunque prometía ser el mas indicado para este uso.

Finalmente luego del proceso de análisis se accede a borrar las reviews pero conservando una estimación de lo expuesto en las reviews, permitiendo mejor tratamiento cuantitativo de los datos y además menor tamaño del archivo para luego ser desplegado en FastAPI y Render. [Lab_PI_01.3.NLP.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.3.NLP.ipynb)

## EDA

En el Análsis Exploratorio de Datos, se realizarán distintos procesos de trabajo para interpretar los datos. 

Principalmente se utilizan y se combinan los datos de las 3 tablas junto al NLP para obtener nueva informacion y posteriormente en el deploy se puedan generar recmendaciones y demas pedidos efectuados. 

El trabajo realizado en esta etapa consiste del análisis gráfico y estadistico de las diferentes variables y sus combinaciones. Esto podría servir, en el caso de una empresa a ayudar a los departamentos de marketing, produccion o publicidad a determinar estrategias de trabajo. Ya que se puede conocer que generos de productos son los mas utilizados, tiempo de uso de los mismos o las recomendaciones que han obtenido.

En este caso se puede ver que en el caso de playtime_forever, el tiempo aunque aparentemete ha sido bien extraido, parece ser muy grande en algunos casos, por lo que debería tenerse en cuenta que este no sea en horas, sino minutos de uso, ya que en algunos casos el uso individual excede lo humanamente posible.

Este trabajo, con todos sus graficos y valores cuantitativos se encuentra en [Lab_PI_01.4.EDA.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.4.EDA.ipynb).

## Tablas y Funciones para FastApi y modelo de Recomendaciones

En esta seccion se generan las tablas y funciones necesarias para hacer los deploys correspondientes para FastAPI y render.

### Tablas

Las tablas necesarias para hacer el despliege y que operen ligeramente en las funciones, se procesan en dos archivos, los cuales corresponden a dos etapas del trabajo:

- La primera consiste en un notebook que procesa los datos para realizar posteriormente las consultas "estadísticas" y obtener los datos producidos por las reviews según el análisis de sentimiento, que se encuentran en el archivo [Lab_PI_01.5.1.Tablas_FastAPI.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.5.1.Tablas_FastAPI.ipynb). La información se almacena en [merged_steam_rev_data.parquet](https://github.com/mdallanegra/LAB_PI_01/blob/main/FastAPI/merged_steam_rev_data.parquet) y en [merged_steam_user_data.parquet](https://github.com/mdallanegra/LAB_PI_01/blob/main/FastAPI/merged_steam_user_data.parquet).
- La segunda es para el modelo de recomendacion, que se procesa en el notebook [Lab_PI_01.5.2.Tablas_Modelo_Recomendacion.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.5.2.Tablas_Modelo_Recomendacion.ipynb) guardando la informacion en [merged_recommend_model.parquet](https://github.com/mdallanegra/LAB_PI_01/blob/main/FastAPI/merged_recommend_model.parquet).

Es muy importante aclarar que debido a las restricciones de memoria y recursos de FastAPI y Render, las tablas se han reducido utilizando un metodo de sampleado aleatorio para que ocupen poca RAM en los procesos.

### Funciones

#### Desarrollo API

Aqui se producen las funciones solicitadas para el framework FastAPI, las cuales serán:

- **def PlayTimeGenre( genero : str ):** -> {"Año de lanzamiento con más horas jugadas para Género 'genre'" : year}
- **def UserForGenre( genero : str ):** -> {"Usuario con más horas jugadas para Género 'genre'" : 'user_id', "Horas jugadas":[{Año: year, Horas: hours}, {Año: year, Horas: hours}, ... ]}
- **def UsersRecommend( año : int ):** -> [{"Puesto 1" : 'item_id'}, {"Puesto 2" : 'item_id'},{"Puesto 3" : 'item_id'}]
- **def UsersNotRecommend( año : int ):** -> [{"Puesto 1" : 'item_id'}, {"Puesto 2" : 'item_id'},{"Puesto 3" : 'item_id'}]
- **def sentiment_analysis( año : int ):** -> {Negativo = 'count', Neutral = 'count', Positive = 'count'}

Estas funciones se encuentran en el notebook [Lab_PI_01.6.Funciones_FastAPI.ipynb](https://github.com/mdallanegra/LAB_PI_01/blob/main/_notebooks/Lab_PI_01.6.Funciones_FastAPI.ipynb), donde se puede ver el proceso de creacion de cada una de ellas.

#### Modelo de Recomendación

En esta sección se encuentran las funciones de recomendacion basadas en cosine-similarity del modulo sklear, que implementan las sugerencias item-item y user-item:

- item-item asigna 5 resultados por similaridad respecto a un item ingresado.
  - **def recomendacion_juego( id de producto ):** -> [{"Juego recomendado 1" : {app_name}}, 
                                                {"Juego recomendado 2" : {app_name}},
                                                ...]
- user-item asigna 5 resultados por similaridad respecto a los gustos del usuario ingresado.
  - **def recomendacion_usuario( id de usuario ):** -> [{"Juego recomendado 1" : {app_name}}, 
                                                 {"Juego recomendado 2" : {app_name}},
                                                 ...]
Por ultimo ambos juegos de funciones anteriormente mencionadas, se encuentrarn en [FastAPI.py](https://github.com/mdallanegra/LAB_PI_01/blob/main/FastAPI.py) con la version definitiva de cada una de ellas, con su docstring correspondiente y listas para hacer el deploy.

## Deploy FastAPI & Render


En esta seccion se implementan las funciones anteriormente generadas y se las despliegan en el framework FastAPI en forma local y en Render en forma pública.

- Localmente se hace el deploy utilizando el comando `uvicorn FastAPI:app --host 0.0.0.0 --port 8080` y
se encontrará en la direccion web de FastAPI [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- En forma pública se utiliza tambien el comando `uvicorn FastAPI:app --host 0.0.0.0 --port 8080` y se encontrará en la direccion web de Render [https://henrylabspi01.onrender.com/docs](https://henrylabspi01.onrender.com/docs). Hay que tener en cuenta que se creó un archivo [requirements.txt](https://github.com/mdallanegra/LAB_PI_01/blob/main/requirements.txt) haciendo `pip freeze > requirements.txt`, con los modulos necesarios para el despliegue.

## Video

Google Drive: [Video Explicativo](video.com)

## Conclusiones

Se puede decir que el trabajo se ha resuelto en su totalidad, aunque con ciertos inconvenientes como por ejemplo para extraer correctamente los datos y para hacer el deploy, pero afortunadamente se han podido solucionar.

Tambien se ha podido notar que las tres tablas no poseen toda la informacion correcpondiente a cada item_id o app_name, por lo que no siempre al unir las bases de datos, se ha logrado obtener la totalidad de la información, por la que para un trabajo posterior estos podrían ser incluidos.

Algunos datos han presentado valores inconsistentes, ya que como se mencionó anteriormente las dimensiones de los datos de playtime_forever, aparentan ser demasiado grandes y probablemente la unidad de medida no sea la que se interpreta a priori.

Finalmente se puede decir que en el NLP se podría utilizar un metodo mas adecuado para obtener un análisis de sentimientos mas exacto, si se usan modelos que entiendan mejor los dobles sentidos, emojis y lenguaje coloquial. Tambien en el caso del modelo de recomendacion, se puede buscar una manera mas robusta de generar la matriz de factorización que además pueda manejar mayor numeros de datos y cree vectores mas exactos para apuntar a los resultados de los diferentes pedidos.

***

## Contacto


<div class="icon-with-text-container">
    <div class="ivm-view-attr__img-wrapper display-flex">
        <li-icon type="email-envelope-icon" class="" size="large" role="img" aria-label="Correo electrónico">
           <i class="fas fa-envelope fa-2x"></i>
        </li-icon>
        <a href="mailto:mdallanegra@icloud.com" class="text-link-next-to-icon"> Correo Electrónico</a>
    </div>    
</div>

<div class="icon-with-text-container">
    <div class="ivm-view-attr__img-wrapper display-flex">
      <li-icon type="app-linkedin-bug-color-icon" class="" size="large" role="img" aria-label="LinkedIn">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -3 24 24" data-supported-dps="24x24" fill="currentColor" class="mercado-match" width="34" height="34" focusable="true"><path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"></path></svg> <a href="https://www.linkedin.com/in/miguel-angel-dallanegra-vilches-b419b19b/" class="text-link-next-to-icon"> Perfil de LinkedIn</a></li-icon>
    </div>
</div>

</body>
</html>