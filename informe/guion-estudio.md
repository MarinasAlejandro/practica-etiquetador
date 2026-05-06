# Guion de estudio — Práctica Etiquetador

Este documento contiene absolutamente todo lo que necesito saber sobre la práctica del etiquetador automático de imágenes de ropa: el contexto, la teoría, lo que implementé, los análisis, los resultados, las limitaciones y las posibles preguntas que me puede hacer el profesor en la oral. Está escrito en prosa para poder convertirlo en un podcast con NotebookLM y estudiarlo escuchando.

## Parte 1: Qué hace el sistema y para qué sirve

El proyecto consiste en construir un agente de etiquetado automático para una tienda online de ropa. La idea es que cuando llegan productos nuevos al catálogo, normalmente sin etiquetas, el sistema los analice solo y les asigne dos tipos de información: la forma de la prenda y los colores predominantes. Sobre estas etiquetas se construye después un buscador combinado, que permite a los usuarios filtrar por consultas tipo "vestido rosa" o "tacones negros" en lenguaje natural.

El sistema asigna nueve categorías de forma: vestidos, camisas, tacones, chanclas, pantalones cortos, vaqueros, calcetines, sandalias y bolsos. Para el color usa once colores básicos universales: rojo, naranja, marrón, amarillo, verde, azul, morado, rosa, negro, gris y blanco.

El dataset con el que se entrena y se evalúa el sistema tiene dos mil trescientas veintiocho imágenes de entrenamiento y ochocientas cincuenta y una de test. Es importante recordar que el enunciado del PDF mencionaba ocho categorías de forma, pero el ground truth real del dataset incluye una categoría extra que es "tacones". El sistema se ha entrenado con las nueve clases reales del archivo de etiquetas, no con las ocho que decía el PDF, porque tiene que funcionar con los datos que hay.

Además del núcleo del proyecto, se ha desarrollado un frontend web con Flask que ofrece dos funcionalidades. La primera permite subir una imagen y recibir su etiqueta de forma y los colores predominantes. La segunda es un buscador que filtra el catálogo por forma y color combinados, mostrando una galería de imágenes coincidentes.

## Parte 2: Conceptos fundamentales que hay que tener claros

Antes de explicar cada parte del sistema, conviene fijar el vocabulario básico de aprendizaje automático.

Clasificar significa asignar una categoría a un elemento nuevo. En este proyecto, clasificar una imagen es decidir si es un vestido, una camisa o cualquier otra de las nueve clases.

Las clases o etiquetas son las categorías posibles. En este caso, las nueve formas de prenda.

El aprendizaje supervisado es cuando le enseñas al modelo con ejemplos que ya están etiquetados. Por ejemplo, mil imágenes de vestidos marcadas como "vestido", mil de camisas marcadas como "camisa", y así sucesivamente. El modelo aprende a partir de ese ground truth.

El aprendizaje no supervisado, en cambio, no tiene etiquetas. El modelo tiene que descubrir patrones por su cuenta. Por ejemplo, agrupar píxeles parecidos sin saber qué es cada color.

El conjunto de entrenamiento, o train, son los ejemplos con etiqueta que el modelo "ve" durante el aprendizaje.

El conjunto de test son imágenes nuevas que el modelo no ha visto durante el entrenamiento. Sirven para evaluar lo bien que generaliza a casos no vistos.

Una característica o feature es un número que describe parte del elemento. En este proyecto, cada uno de los píxeles en escala de grises de la imagen es una característica.

El espacio de características es el "mundo" matemático donde viven los datos. Si una imagen se describe con cuatro mil ochocientos píxeles, vive en un espacio de cuatro mil ochocientas dimensiones.

La distancia euclidiana es la forma más habitual de medir cuán parecidos son dos puntos en ese espacio. Es la "distancia en línea recta" generalizada a muchas dimensiones. Se calcula como la raíz cuadrada de la suma de las diferencias al cuadrado de cada coordenada. Distancia pequeña significa puntos parecidos. Distancia grande significa puntos diferentes.

## Parte 3: KNN para clasificar la forma

KNN significa "K vecinos más cercanos", de las siglas en inglés "K Nearest Neighbours". Es un algoritmo de clasificación supervisada muy intuitivo y conceptualmente simple.

La idea es la siguiente: el modelo guarda todas las imágenes de entrenamiento con sus etiquetas. Cuando llega una imagen nueva que hay que clasificar, calcula la distancia entre esa imagen y todas las del entrenamiento. Después ordena esas distancias de menor a mayor, coge las k más cercanas, mira las etiquetas de esos k vecinos y vota la clase mayoritaria. La clase que más aparezca entre los vecinos es la predicción.

El parámetro k es muy importante. Si k es uno, te quedas con el vecino más cercano y ya. Si k es tres, miras los tres más cercanos y votas. Con k pequeño el modelo es muy sensible a casos raros u outliers, porque un solo vecino mal etiquetado puede contaminar la decisión. Con k grande, el modelo es más robusto pero pierde detalle local.

En este proyecto se usa k igual a tres como recomendación de producción, aunque en los experimentos se ha probado k igual a uno, tres, cinco, siete, nueve y once.

Antes de aplicar KNN hay que preprocesar las imágenes. Cada imagen viene en formato ochenta por sesenta píxeles a color, con tres canales RGB. Es decir, ochenta por sesenta por tres números, que son catorce mil cuatrocientos valores por imagen. El preprocesado hace dos cosas. Primero, convertir la imagen a escala de grises usando una función proporcionada en el código del template, llamada rgb2gray, que combina los tres canales en uno solo aplicando los pesos clásicos de luminosidad. Después, aplanar la imagen, es decir, convertir la matriz de ochenta por sesenta en un vector unidimensional de cuatro mil ochocientos números. Cada número es un píxel y cada píxel es una característica.

¿Por qué se pasa a gris para clasificar la forma? Porque para distinguir una camiseta de un vestido o de unos pantalones, el color es ruido. Una camiseta roja y una camiseta azul tienen la misma silueta. Pasar de tres canales a uno solo elimina información irrelevante para la tarea de clasificar la forma, simplifica el espacio de características de catorce mil cuatrocientas dimensiones a cuatro mil ochocientas, y de hecho mejora ligeramente el accuracy porque el modelo deja de "distraerse" con el color.

Una vez preprocesadas las imágenes, el algoritmo KNN procede así. Para cada imagen de test, calcula la distancia euclidiana entre esa imagen y las dos mil trescientas veintiocho del train. Ordena las distancias, coge las k más pequeñas, mira las etiquetas de esos vecinos y devuelve la clase mayoritaria. Todo esto se hace con numpy básico, sin usar librerías como scikit-learn, siguiendo la teoría del Bloque cuatro de la asignatura.

## Parte 4: K-Means para detectar los colores

K-Means es un algoritmo de aprendizaje no supervisado, concretamente un algoritmo de clustering. Clustering significa agrupar datos similares sin tener etiquetas previas. El objetivo es descubrir patrones por sí solo.

La idea de K-Means es la siguiente. Tienes un conjunto de puntos en un espacio. Decides cuántos grupos quieres encontrar, ese número se llama K. El algoritmo coloca K centroides, que son puntos que representarán el centro de cada grupo, y después itera asignando cada punto al centroide más cercano y recalculando la posición de los centroides como la media de los puntos asignados a cada uno. Repite hasta que los centroides ya no se mueven, lo que se llama convergencia.

En este proyecto, K-Means se usa para detectar los colores predominantes de cada imagen. La idea clave es tratar cada píxel como un punto en el espacio RGB de tres dimensiones. Una imagen de ochenta por sesenta tiene cuatro mil ochocientos píxeles, así que es como tener cuatro mil ochocientos puntos en un espacio tridimensional. K-Means agrupa esos puntos en K clusters, y los centroides resultantes son los colores predominantes.

El algoritmo en mi implementación funciona así. Primero, inicialización. Se eligen K centroides iniciales con una de dos estrategias. La estrategia "first" coge los K primeros píxeles distintos de la imagen. La estrategia "random" coge K píxeles aleatorios. Ambas estrategias son válidas y dan resultados muy parecidos.

Segundo, asignación. Para cada píxel se calcula su distancia euclidiana a los K centroides y se asigna al más cercano. Esto se hace de forma vectorizada con numpy.

Tercero, recálculo. Para cada cluster, el nuevo centroide es la media de todos los píxeles asignados a él. Si un cluster se queda vacío, se mantiene el centroide anterior.

Cuarto, comprobación de convergencia. Si los centroides no se han movido respecto a la iteración anterior, dentro de una tolerancia, el algoritmo para. Si no, se vuelve al paso de asignación. Hay un máximo de iteraciones para evitar que el bucle no termine nunca, fijado en cien.

Una vez convergido, los K centroides representan los K colores predominantes de la imagen, expresados como vectores RGB.

## Parte 5: Encontrar la K óptima con find_bestK

Un problema importante de K-Means es que hay que decirle de antemano cuántos clusters K quieres encontrar. Pero en una imagen no sabes a priori si tiene tres, cinco o siete colores predominantes. La práctica pide un mecanismo automático para decidir K, llamado find_bestK.

El criterio que pide el PDF en la página ocho es el siguiente. Se calcula una métrica llamada distancia intra-clase, abreviada WCD por "within class distance". La WCD es la media de las distancias al cuadrado de cada píxel a su centroide. Mide cómo de compactos son los clusters: si la WCD es baja, los puntos están muy pegados a sus centros, lo que es bueno. Si es alta, los puntos están dispersos.

Naturalmente, cuanto más K, menor WCD, porque al haber más clusters cada cluster es más pequeño y más compacto. Pero pasado cierto K, añadir un cluster más ya no aporta gran mejora. Ese es el punto óptimo.

El criterio del PDF dice: prueba K desde dos hasta un máximo, calcula la WCD para cada K, y mira el porcentaje de mejora respecto al K anterior. La fórmula es cien menos cien por la WCD del K actual dividido entre la WCD del K anterior. Si ese porcentaje es menor del veinte por ciento, significa que añadir ese cluster ya no compensa. Te quedas con el K anterior.

Por ejemplo, si con K igual a dos la WCD es cuarenta y nueve, con K igual a tres es veintinueve, con K igual a cuatro es veintiocho, el decremento de cuatro a tres es de cuarenta y un por ciento, claramente significativo. El decremento de cuatro a tres respecto al anterior es de un cuatro por ciento, mucho menor que el veinte. Por tanto, K igual a tres es la elección óptima en ese ejemplo.

## Parte 6: De RGB a nombre de color con get_colors

Una vez K-Means ha detectado los colores predominantes, esos colores son vectores RGB de tres números. Pero el sistema tiene que devolver nombres de colores legibles, como "rosa" o "azul". Para eso se usa la función get_color_prob, que está proporcionada en el archivo utils.py del template y que no hay que implementar.

Esta función recibe un color RGB y devuelve un vector de once probabilidades, una por cada color básico universal: rojo, naranja, marrón, amarillo, verde, azul, morado, rosa, negro, gris y blanco. La probabilidad indica cuán parecido es el color RGB a cada uno de los once nombres.

Mi función get_colors recibe los centroides RGB de K-Means, los pasa a get_color_prob para obtener las once probabilidades de cada uno, y se queda con el nombre del color con mayor probabilidad para cada centroide. Es lo que se llama un argmax.

Por ejemplo, si un centroide es el RGB doscientos cuarenta y uno, ciento ochenta y cuatro, noventa y cuatro, get_color_prob devuelve un vector donde la probabilidad de "amarillo" es noventa y cuatro por ciento. El argmax es la posición del amarillo, así que get_colors devuelve "Yellow".

## Parte 7: El sistema completo y el buscador

Una vez tengo los dos modelos por separado, los junto en una función llamada predict_image que es el etiquetador end-to-end. Recibe una imagen suelta y devuelve un diccionario con la forma predicha y la lista de colores predominantes. Internamente, hace lo siguiente. Primero aplica KNN para predecir la forma, usando k igual a tres. Después aplica K-Means con find_bestK para decidir automáticamente cuántos colores hay, y get_colors para obtener los nombres. Devuelve forma, colores y el K efectivamente usado.

Sobre el buscador. Hay tres funciones de retrieval. La primera, retrieval_by_color, devuelve las imágenes cuyas etiquetas de color contienen el color buscado. Las etiquetas de color son listas, porque una imagen puede tener varios colores predominantes. La segunda, retrieval_by_shape, devuelve las imágenes cuya etiqueta de forma coincide exactamente con la buscada. La tercera, retrieval_combined, hace un AND lógico de las dos anteriores: devuelve las imágenes que cumplen color y forma a la vez.

Para que el buscador acepte queries en lenguaje natural, se ha implementado un mapeo de sinónimos. El usuario puede escribir "dress" o "dresses", y el sistema lo normaliza a la etiqueta del dataset que es "Dresses". Lo mismo para colores: acepta cualquier capitalización.

Un punto importante es que el buscador opera sobre las etiquetas predichas por nuestro propio sistema, no sobre el ground truth. Esto simula el escenario real de una tienda online que recibe productos sin etiquetas, los procesa con el etiquetador, y permite buscarlos. Para que el frontend funcione rápido, hay un script llamado preprocess_dataset que pre-etiqueta las tres mil ciento setenta y nueve imágenes del dataset entero y guarda el resultado en un archivo JSON que el servidor carga en memoria al arrancar.

## Parte 8: Frontend con Flask

El frontend se ha implementado con Flask, un microframework de Python para web. Tiene tres endpoints relevantes. El home, que sirve la página HTML con las dos secciones del frontend. El endpoint predict, que recibe una imagen subida por el usuario, la procesa con predict_image y devuelve un JSON con la forma y los colores. Y el endpoint search, que recibe parámetros de filtrado por color y forma, busca en el JSON pre-etiquetado y devuelve la lista de imágenes coincidentes.

La interfaz HTML tiene dos secciones. La primera permite subir una imagen y muestra el resultado debajo. La segunda permite seleccionar una forma y un color de menús desplegables, y al pulsar "Buscar" muestra una galería de imágenes del catálogo que cumplen los filtros.

Un detalle técnico es que la app corre en el puerto cinco mil uno en lugar del clásico cinco mil. Esto es porque en macOS, desde la versión Monterey, el puerto cinco mil está ocupado por el AirPlay Receiver del sistema y bloquea cualquier servidor local que quiera usarlo.

## Parte 9: Los cuatro análisis de eficiencia

El enunciado pide un mínimo de tres análisis de parámetros en el informe. Yo hice cuatro para cubrir distintos aspectos del sistema.

El primer análisis compara las dos estrategias de inicialización de K-Means: "first" y "random". Se ejecuta sobre cincuenta imágenes con K igual a cuatro y se mide el número de iteraciones hasta convergencia, la WCD final y el tiempo. El resultado es que "random" converge un diez por ciento más rápido en iteraciones, en concreto veintiuno coma cero seis iteraciones de media frente a veintitrés coma cinco con "first", y la WCD final es prácticamente igual. La conclusión es que ambas estrategias son válidas y la diferencia es marginal. Yo uso "first" por defecto porque es determinista, es decir, da el mismo resultado siempre con la misma imagen, y eso facilita la depuración.

El segundo análisis mide el accuracy del KNN sobre las ochocientas cincuenta y una imágenes de test para varios valores de k. Con k igual a uno obtengo noventa y uno coma ochenta y nueve por ciento de accuracy. Con k igual a tres bajo a noventa coma veinticinco por ciento. Con k igual a cinco a ochenta y nueve coma cero siete. Con k igual a siete sube ligeramente a ochenta y nueve coma cuarenta y dos. Con k igual a nueve baja a ochenta y ocho coma cero uno. Con k igual a once a ochenta y siete coma setenta y ocho. La tendencia general es decreciente al aumentar k, con un pequeño repunte en k igual a siete. Esto sugiere que las clases están bien separadas en el espacio de píxeles: los vecinos más cercanos suelen ser de la misma clase, así que añadir más vecinos solo dispersa la decisión. La recomendación de producción es k igual a tres porque mantiene un accuracy muy alto pero es más robusto a outliers que k igual a uno.

El tercer análisis estudia el criterio de find_bestK. Compara distintos umbrales del decremento de WCD: diez por ciento, quince, veinte, veinticinco y treinta. Sobre treinta imágenes elegidas al azar. A mayor umbral, menor K elegida. Con un umbral del diez por ciento, la K media es seis coma noventa, es decir, detecta entre seis y siete colores por imagen. Con el veinte por ciento, que es el del PDF, la K media baja a cinco coma seis. Con el treinta por ciento, baja a cuatro coma cuatro. La conclusión es que el veinte por ciento del PDF es un equilibrio razonable: detecta entre tres y siete colores por imagen, suficiente para representar los colores principales del producto más algunos del fondo.

El cuarto análisis es sobre la normalización Min-Max en el KNN. La teoría del Bloque cuatro, sesión siete, dice que el KNN decide solo a partir de distancias, así que la escala de las variables importa: una variable con valores grandes puede dominar el cálculo. La normalización Min-Max transforma los valores al rango cero uno usando la fórmula valor menos mínimo dividido entre máximo menos mínimo. El análisis compara el accuracy con y sin normalización. El resultado es que el accuracy es exactamente el mismo en ambos casos, noventa coma veinticinco por ciento. Esto puede parecer contradictorio con la teoría a primera vista, pero tiene una explicación clara. Las cuatro mil ochocientas características son píxeles en escala de grises, todos en el mismo rango cero a doscientos cincuenta y cinco. La normalización divide todo por la misma constante, lo que es equivalente a multiplicar todas las distancias por la misma constante. Como KNN solo mira el orden de las distancias, multiplicar por una constante no cambia ese orden. La normalización es fundamental cuando hay variables con escalas distintas, pero es neutra cuando todas las variables ya están en el mismo rango.

## Parte 10: Resultados y matriz de confusión

El accuracy global del sistema con k igual a tres es noventa coma veinticinco por ciento sobre las ochocientas cincuenta y una imágenes de test. Pero ese número global esconde matices importantes que aparecen al hacer una matriz de confusión por clase.

Las clases que mejor funcionan son los bolsos con un noventa y seis coma ocho por ciento de accuracy, los pantalones vaqueros también con noventa y seis coma ocho, las camisas con noventa y cinco coma nueve, y los vestidos con noventa y tres coma cinco. Estas son categorías visualmente muy distintas del resto, así que el modelo apenas se confunde.

Las clases que peor funcionan son los tacones con setenta y siete coma ocho por ciento, las sandalias con ochenta y uno coma nueve, y las chanclas con ochenta y ocho coma siete. Casualmente, son los tres tipos de calzado abierto.

Mirando dónde se confunden, el patrón es muy claro. El diecisiete por ciento de los tacones se predicen como chanclas. El dieciséis por ciento de las sandalias se predicen como chanclas. Y un seis por ciento de las chanclas se predicen como sandalias. Las únicas confusiones graves se dan entre estos tres tipos de calzado, que visualmente son casi idénticos en una foto pequeña en escala de grises. Un humano también dudaría.

Lo importante es que entre clases muy distintas, como calcetines y pantalones, o vestidos y camisas, o bolsos y cualquier otra cosa, el sistema no comete errores significativos. Esto demuestra que las clases visualmente diferentes están bien separadas en el espacio de píxeles.

## Parte 11: Limitaciones y riesgos del sistema

El sistema funciona bien con el dataset de la práctica pero conviene ser honestos con los casos en los que falla.

Primero, las imágenes con fondo no uniforme. Las del catálogo tienen fondos blancos o muy uniformes. Si se sube una foto en la calle con fondo lleno de detalles, K-Means detectaría los colores del fondo como predominantes, y el KNN se confundiría porque la silueta no quedaría aislada.

Segundo, las prendas no incluidas en el entrenamiento. El KNN solo conoce las nueve clases. Una imagen de un abrigo, un cinturón o un sombrero se clasificaría forzosamente como la clase más visualmente parecida, sin que el sistema sepa que se equivoca. No hay un mecanismo de "no sé".

Tercero, los píxeles del producto frente a los del fondo. K-Means analiza toda la imagen sin distinguir el producto del fondo. Si una camisa azul está sobre un fondo blanco, dirá que el producto es azul y blanco, aunque el blanco sea solo el fondo.

Cuarto, la sensibilidad a la escala y la pose. Como el KNN compara píxel a píxel, dos imágenes de la misma prenda en posiciones distintas, como centrada o descentrada, grande o pequeña, pueden quedar lejos en el espacio aunque sean idénticas para un humano.

Quinto, el criterio del veinte por ciento en find_bestK es heurístico. Funciona en la mayoría de casos pero no es óptimo: en imágenes muy uniformes elige K demasiado pequeñas, y en imágenes con muchos detalles elige K demasiado grandes.

En todos estos casos el sistema devuelve una respuesta sin avisar de que duda, así que confiar ciegamente en sus etiquetas para automatizar decisiones puede llevar a errores silenciosos.

## Parte 12: Contexto de aplicación

¿Dónde tiene sentido este sistema? En el catálogo de una tienda online controlado, donde las imágenes llegan con un formato homogéneo: fondo neutro, prenda centrada, tamaño consistente. Ahí un noventa por ciento de accuracy es un buen punto de partida para automatizar el etiquetado y dejar que un humano revise solo los casos dudosos. También en sistemas de búsqueda interna donde un fallo no tiene consecuencias graves: si el usuario busca "vestido rosa" y se le cuela un vestido salmón, no pasa nada, puede refinar la búsqueda.

¿Dónde no usaría este sistema? En aplicaciones donde un error tiene coste real, como la devolución automática de productos según etiquetas predichas o decisiones de stock. Un noventa por ciento significa que uno de cada diez productos quedaría mal categorizado, lo que se acumula con miles de productos. Tampoco en cualquier dominio crítico, médico, de seguridad o de decisiones sobre personas. El sistema no está diseñado para eso, no se ha validado con cross-validation y no maneja incertidumbre.

## Parte 13: Posibles preguntas del profesor en la oral

El profesor solo va a mirar el informe y probar el frontend, no va a auditar el código. Pero puede hacer preguntas sobre los conceptos, las decisiones y los resultados. Estas son las preguntas más probables y la forma de contestarlas.

Si pregunta qué hace el sistema, responder: es un etiquetador automático que dada una imagen de ropa predice la forma con KNN supervisado y los colores predominantes con K-Means no supervisado, y un buscador que combina ambas etiquetas para resolver consultas en lenguaje natural.

Si pregunta por qué dos modelos en vez de uno, responder: porque resuelven problemas distintos. La forma es supervisada porque hay etiquetas, así que se usa KNN. El color es no supervisado porque no se sabe a priori cuántos colores hay en cada imagen, así que se usa K-Means para descubrirlos.

Si pregunta por qué se pasa a gris para clasificar la forma, responder: porque para distinguir la silueta de una prenda el color es ruido. Una camiseta roja y una azul tienen la misma forma. Reducir tres canales a uno elimina información irrelevante y simplifica el espacio de características.

Si pregunta qué k se ha usado y por qué, responder: en producción se usa k igual a tres. K igual a uno daba el mejor accuracy con noventa y uno coma ochenta y nueve por ciento, pero es muy sensible a outliers. Con k igual a tres se mantiene un noventa coma veinticinco por ciento y es más robusto.

Si pregunta cómo se elige el número de colores en K-Means, responder: con find_bestK. Se prueba K desde dos hasta un máximo, se calcula la WCD que es la distancia intra-clase, y cuando añadir un cluster reduce la WCD menos del veinte por ciento, se para. Es el criterio del PDF.

Si pregunta cómo se convierte un centroide RGB en un nombre de color, responder: con la función get_color_prob proporcionada en utils.py, que devuelve un vector de once probabilidades, una por cada color básico universal. La función get_colors coge el argmax de esas probabilidades.

Si pregunta por la matriz de confusión, responder: las únicas confusiones graves se dan entre los tres tipos de calzado abierto, tacones, sandalias y chanclas, que son visualmente casi idénticos en escala de grises. El diecisiete por ciento de los tacones se predicen como chanclas. Pero entre categorías muy distintas como calcetines, vestidos, pantalones o bolsos, el sistema no comete errores graves.

Si pregunta por las limitaciones, responder con los cinco puntos clave: fondos no uniformes, prendas fuera del entrenamiento, fondo contaminando los colores, sensibilidad a pose y escala, y heuristicidad del criterio del veinte por ciento.

Si pregunta cuándo no usaría este sistema, responder: en aplicaciones con coste real de errores, como decisiones automáticas de stock o devoluciones, o en dominios críticos como medicina o seguridad.

Si pregunta por qué se hicieron cuatro análisis y no tres, responder: el enunciado pide mínimo tres pero hice cuatro para cubrir distintos aspectos del sistema. Uno del K-Means con la inicialización, uno del KNN con el valor de k, uno del find_bestK con el criterio del veinte por ciento, y uno cruzado con la normalización Min-Max.

Si pregunta por qué la normalización no cambia el accuracy, responder: porque todas las características son píxeles en el mismo rango cero a doscientos cincuenta y cinco. La normalización Min-Max divide todo por la misma constante y como KNN solo mira el orden de las distancias, no cambia los rankings de vecinos. La normalización es clave cuando hay variables con escalas distintas, no aquí.

Si pregunta cuántas categorías de forma hay, responder: nueve. El PDF mencionaba ocho pero el ground truth real del dataset incluye una novena que son los tacones. El sistema se ha entrenado con las nueve reales.

Si pregunta por la presentación de la mañana al ver los timestamps de los commits, responder con honestidad: trabajé hasta tarde porque tenía la deadline al día siguiente, y por la mañana hice algunas correcciones después de revisar el informe.

## Parte 14: Cierre

Este sistema cumple los tres objetivos del enunciado, etiquetado de forma con KNN, etiquetado de color con K-Means y buscador combinado, y añade un frontend Flask para hacer la práctica más útil. El KNN alcanza un noventa por ciento de accuracy, K-Means detecta correctamente los colores predominantes y find_bestK con el criterio del veinte por ciento del PDF da una K media razonable. La parte más interesante del proyecto ha sido entender que la representación de los datos importa tanto como el algoritmo: la decisión del PDF de pasar a gris no es un detalle menor, es lo que hace que el KNN funcione bien y que la normalización Min-Max no aporte nada porque ya garantiza misma escala. También ha sido útil ver cómo dos algoritmos básicos, uno supervisado y otro no supervisado, se combinan para resolver un problema real cuando uno solo no podría.
