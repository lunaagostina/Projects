#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nombre del grupo: CPQ.
Estudiantes:Camila Molteni Ceccón
            Luna Praino
            Agustin Quintriqueo
            
Descripción: en este archivo se encuentra el código del respectivo trabajo, es 
            decir, el analisis exploratorio, los modelos, los resultados obteni
            dos,etc.
            Además se encuentra el código realizado por los profesores para po
            der visualizar las letras del dataset. El mismo lo mantuvimos en 
            nuestro archivo por razones de prolijidad y también para mayor 
            comprensión del Data frame.
"""
encoding= 'UTF-8'
#%%
# Sección de imports.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
#%%
#Sección de carga de datos.
data = pd.read_csv("/Users\\camil\\Documents\\Laboratorio_de_Datos\\TP02-CPQ\\\\emnist_letters_tp.csv", header= None)
#%%
#Funciones que utilizamos.
#Rota las imagenes de las letras del dataframe.
def flip_rotate(image):
    """
    Función que recibe un array de numpy representando una
    imagen de 28x28. Espeja el array y lo rota en 90°.
    """
    W = 28
    H = 28
    image = image.reshape(W, H)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

#Esta función devuelve cuántos ceros(negro),blancos(255) y 
#grises(del 0 sin incluir al 255 sin incluir) hay en el dataframe ingresado.
def tonos (data):
    negros=0 
    grises=0 
    blancos=0
    for line in data.itertuples():
        for elemento in line:
            if elemento == 0:
                negros+=1
            elif elemento ==255:
                blancos+=1
            else:
                grises+=1
    return negros,blancos,grises

#Esta función devuelve el promedio de cada pixel del dataframe ingresado.
def promedio_pixeles(df):
    promedios = df.mean(axis=0).values
    promedio_imagen = promedios.reshape((28,28))
    return promedio_imagen

#Esta función cuenta la cantidad de componentes en la clase A y en la clase L.
def balanceado (data):
    cant_L = 0
    cant_A = 0 
    for line in data.itertuples():
            if line[1] == 'L':
                cant_L+=1
            else:
                cant_A+=1
    return cant_L,cant_A

# Muestra las imágenes con los píxeles relevantes superpuestos
def mostrar_superposicion(imagen_letra, imagen_pixeles, titulo):
    fig, ax = plt.subplots()

    # Mostrar la imagen de la letra promedio
    ax.imshow(imagen_letra, cmap='gray')

    # Superponer los píxeles relevantes con transparencia
    ax.imshow(imagen_pixeles, cmap='gray', alpha=0.5)

    ax.set_title(titulo)
    plt.axis('off')

    plt.show()
    
def matriz_confusion_binaria(y_test, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_test)):
        if y_test[i]:
            if y_pred[i]:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i]:
                fp += 1
            else:
                tn += 1
    
    return tp, tn, fp, fn



#%%
#Codigo que nos dieron los profesores.
# Elijo la fila correspondiente a la letra que quiero graficar
n_row = 14
row = data.iloc[n_row].drop(0)
letra = data.iloc[n_row][0]

image_array = np.array(row).astype(np.float32)

# Ploteo el grafico
plt.imshow(image_array.reshape(28, 28))
plt.title('letra: ' + letra)
plt.axis('off')  
plt.show()

# Se observa que las letras estan rotadas en 90° y espejadas

#Aca se utiliza la función flip_rotate.
# Ploteo la imagen transformada
plt.imshow(flip_rotate(image_array))
plt.title('letra: ' + letra)
plt.axis('off')  
plt.show()

#Comparamos con otra letra.
n_row = 8
row = data.iloc[n_row].drop(0)
letra = data.iloc[n_row][0]

image_array = np.array(row).astype(np.float32)

# Ploteo el grafico
plt.imshow(image_array.reshape(28, 28))
plt.title('letra: ' + letra)
plt.axis('off')  
plt.show()

# Se observa que las letras estan rotadas en 90° y espejadas

#Aca se utiliza la función flip_rotate.
# Ploteo la imagen transformada
plt.imshow(flip_rotate(image_array))
plt.title('letra: ' + letra)
plt.axis('off')  
plt.show()

#%%
#-------------Primer Punto--------------------
#
#%%
#Item A.
#Uso la funcion tonos para saber que cuántos valores cero, doscientos cincuenta y cinco hay
# y también para saber los valores en medio de ese rango. 
#Para nosotros exiten tres tipos de valores negro(cero),blanco(255) y grises(del 1 hasta el 254).

#Armo los distintos dataset. La idea aca fue primero contar la cantidad de los tres valores menciona
#dos antes para todo el dataframe y luego ir dividiendolo en sectores (conjunto de filas) para averi
#guar donde habian mas valores de blancos y grises.

#Todas las filas
tono = np.array(tonos(data))
grafico = pd.DataFrame(tono)

#De la fila 1 a la 4
tono4 = np.array(tonos(data.iloc[:,1:112]))
grafico4 = pd.DataFrame(tono4)

#De la fila 4 hasta la 24
tono2 = np.array(tonos(data.iloc[:,112:672]))
grafico2 = pd.DataFrame(tono2)

#De la fila 24 hasta la 28 
tono3 = np.array(tonos(data.iloc[:,672:784]))
grafico3 = pd.DataFrame(tono3)




#%%
#Graficos del item A. 
#Creo los graficos para el item A de cada dataset.
#Todo el dataset.
fig, ax = plt.subplots()
#ax.pie(data=grafico, x=grafico[0])
ax.pie(data=grafico, 
       x=grafico[0], 
       labels=['negro(0)','blancos(255)','grises(del 1 al 254)'],          
       autopct='%1.2f%%',       # porcentajes
       colors=['#96E5E4',
               'purple', 'pink'],
       shadow = True,        
       )
ax.set_title('TODAS LAS FILAS DE TODAS LAS LETRAS DEL DATAFRAME')

#De la fila 1 a la 4.
fig, ax = plt.subplots()
#ax.pie(data=grafico, x=grafico[0])
ax.pie(data=grafico4, 
       x=grafico4[0], 
       labels=['negro(0)','blancos(255)','grises(del 1 al 254)'],          
       autopct='%1.2f%%',       # porcentajes
       colors=['#96E5E4',
               'purple', 'pink'],
       shadow = True,         
       )
ax.set_title('FILA 1 A LA 4')

#De las filas 4 hasta la 24.
fig, ax = plt.subplots()
#ax.pie(data=grafico, x=grafico[0])
ax.pie(data=grafico2, 
       x=grafico2[0], 
       labels=['negro(0)','blancos(255)','grises(del 1 al 254)'],         
       autopct='%1.2f%%',       # porcentajes
       colors=['#96E5E4',
               'purple', 'pink'],
       shadow = True,         # separa las slices del pie plot
       )
ax.set_title('FILAS DE LA 4 HASTA LA 24')



#De la fila 24 a la 28.
fig, ax = plt.subplots()
#ax.pie(data=grafico, x=grafico[0])
ax.pie(data=grafico3, 
       x=grafico3[0], 
       labels=['negro(0)','blancos(255)','grises(del 1 al 254)'],         
       autopct='%1.2f%%',       # porcentajes
       colors=['#96E5E4',
               'purple', 'pink'],
       shadow = True,         # separa las slices del pie plot
       )
ax.set_title('FILA 24 A LA 28')

#%%
#Item B.
#Creamos los subdataset que utilizaremos para el item.
l = data.loc[data[0] == 'L']
l = l.iloc[:, 1:]

e = data.loc[data[0] == 'E']
e = e.iloc[:, 1:]

m = data.loc[data[0] == 'M']
m = m.iloc[:, 1:]

#Aca utilizamos la función promedio_pixel para obtener la L, la E y la M promedio. 
p_l = flip_rotate(promedio_pixeles(l)) 
p_e = flip_rotate(promedio_pixeles(e))
p_m = flip_rotate(promedio_pixeles(m)) 

#Creamos los gráficos del resultado devuelto anteriormente.
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(p_e, cmap='gray')
axs[0].set_title('Promedio de E')
axs[0].axis('off')

axs[1].imshow(p_l, cmap='gray')
axs[1].set_title('Promedio de L')
axs[1].axis('off')

axs[2].imshow(p_m, cmap='gray')
axs[2].set_title('Promedio de M')
axs[2].axis('off')


plt.show()


#Calculamos los módulos de las diferencias entre l, m y e.

l_menos_e = np.abs(p_l - p_e) 
l_menos_m = np.abs(p_l - p_m)
e_menos_m = np.abs(p_e - p_m)

#Hacemos los gráficos de las diferencias.
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(l_menos_e, cmap='gray')
axs[0].set_title('Diferencia entre L y E promedios')
axs[0].axis('off')

axs[1].imshow(l_menos_m, cmap='gray')
axs[1].set_title('Diferencia entre L y M promedios')
axs[1].axis('off')

axs[2].imshow(e_menos_m, cmap='gray')
axs[2].set_title('DIferencia entre E y M promedios')
axs[2].axis('off')


#%%
#Item C.
#Creamos el subdataset con solo la clase C.
solo_C =data.loc[data[0]=='C']
solo_C = solo_C.iloc[:, 1:]

#Al subdataset les aplicamos la funcion promedio_pixeles para obtener la C promedio.
p = promedio_pixeles(solo_C) 

# Ploteo la imagen transformada
plt.imshow(flip_rotate(p), cmap = 'gray')
plt.title('letra: C')
plt.axis('off')  
plt.show()

flip_rotate(p)

#%%
#-------------Segundo Punto--------------------
#
#%% 
#Item A.
#Armo el subdataset.
solo_L_A = data.loc[(data[0] == 'L') | (data[0] == 'A')]

#%%
#Item B.
#Veo si las clases del dataframe estan balanceadas, es decir si hay la misma cantidad de L que de A.
#Para eso utilizo la función balanceado con el dataframe solo_L_A.

print('Cantidad de componentes la clase L y A',balanceado(solo_L_A))

#Conclusion si esta balanceada pues es la misma cantidad.

#%%
#Item C y D.
#Digo quienes son mis X e Y.
X = solo_L_A.drop(0,axis = 1)
Y = solo_L_A[0]

#Ahora de los X que separe (que son todos los atributos posibles), selecciono 3 que sean relevantes para ajustarlo
#al modelo de KNN 

X1 = X.iloc[:, [117, 675, 405]] 

#Ahora separo los datos en conjunto de test y de train 
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y, test_size = 0.3) # 70% para train y 30% para test

#MODELO 1 / 3 ATRIBUTOS
model = KNeighborsClassifier(n_neighbors = 8) # modelo en abstracto
model.fit(X_train1, Y_train1) # entreno el modelo con los datos X_train e Y_train
Y_pred1 = model.predict(X_test1) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test1, Y_pred1))
metrics.confusion_matrix(Y_test1, Y_pred1)

#MODELO 2 / 3 ATRIBUTOS
X2 = X.iloc[:, [111,695,391]]

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 8) # modelo en abstracto
model.fit(X_train2, Y_train2) # entreno el modelo con los datos X_train e Y_train
Y_pred2 = model.predict(X_test2) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test2, Y_pred2))
metrics.confusion_matrix(Y_test2, Y_pred2)

#MODELO 3 / 5 ATRUBUTOS 
X3 = X.iloc[:, [111,135,643,695,405]] 

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 8) # modelo en abstracto
model.fit(X_train3, Y_train3) # entreno el modelo con los datos X_train e Y_train
Y_pred3 = model.predict(X_test3) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test3, Y_pred3))
metrics.confusion_matrix(Y_test3, Y_pred3)

#MODELO 4 / 9 ATRIBUTOS 
X4 = X.iloc[:, [111,117,135,391,405,419,643,675,695]] 

X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X4, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 8) # modelo en abstracto
model.fit(X_train4, Y_train4) # entreno el modelo con los datos X_train e Y_train
Y_pred4 = model.predict(X_test4) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test4, Y_pred4))
metrics.confusion_matrix(Y_test4, Y_pred4)

#%%
#Item E 

#MODELO 5 / 6 ATRIBUTOS / 5 VECINOS
X5 = X.iloc[:, [111,117,135,645,675,695]]

X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X5, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 5) # modelo en abstracto
model.fit(X_train5, Y_train5) # entreno el modelo con los datos X_train e Y_train
Y_pred5 = model.predict(X_test5) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test5, Y_pred5))
metrics.confusion_matrix(Y_test5, Y_pred5)

#MODELO 6 / 12 ATRIBUTOS / 12 VECINOS 
X6 = X.iloc[:, [59, 69, 79, 391, 405, 419, 643, 675, 695, 507, 521, 535]]

X_train6, X_test6, Y_train6, Y_test6 = train_test_split(X6, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 12) # modelo en abstracto
model.fit(X_train6, Y_train6) # entreno el modelo con los datos X_train e Y_train
Y_pred6 = model.predict(X_test6) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test6, Y_pred6))
metrics.confusion_matrix(Y_test6, Y_pred6)

#MODELO 7 / 16 ATRIBUTOS / 20 VECINOS 
X7 = X.iloc[:, [59, 69, 79, 111, 117, 135, 279, 299, 391, 405, 419, 507, 521, 535, 643, 675]]

X_train7, X_test7, Y_train7, Y_test7 = train_test_split(X7, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 20) # modelo en abstracto
model.fit(X_train7, Y_train7) # entreno el modelo con los datos X_train e Y_train
Y_pred7 = model.predict(X_test7) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test7, Y_pred7))
metrics.confusion_matrix(Y_test7, Y_pred7) 

#%%
# Visualizacion previa de los atributos (pixeles) seleccionados.

# Crear una imagen de 28x28 con todos los píxeles en blanco (valor 0)
image_3 = np.zeros((28, 28))

# Píxeles relevantes (3)
pixels_3 = [(4, 14), (14, 4), (24, 24)]

for pixel in pixels_3:
    image_3[pixel] = 255

# Mostrar la imagen
plt.imshow(image_3, cmap='gray')
plt.title("Píxeles relevantes (3) para distinguir 'A' de 'L'")
plt.show()


# Crear una imagen de 28x28 con todos los píxeles en blanco (valor 0)
image_3_case2 = np.zeros((28, 28))

# Píxeles relevantes (3) - Caso 2
pixels_3_case2 = [(4, 4), (14, 14), (24, 14)]

for pixel in pixels_3_case2:
    image_3_case2[pixel] = 255

# Mostrar la imagen
plt.imshow(image_3_case2, cmap='gray')
plt.title("Píxeles relevantes (3) - Caso 2 para distinguir 'A' de 'L'")
plt.show()


# Crear una imagen de 28x28 con todos los píxeles en blanco (valor 0)
image_5 = np.zeros((28, 28))

# Píxeles relevantes (5)
pixels_5 = [(4, 4), (4, 14), (14, 4), (14, 14), (24, 4)]

for pixel in pixels_5:
    image_5[pixel] = 255

# Mostrar la imagen
plt.imshow(image_5, cmap='gray')
plt.title("Píxeles relevantes (5) para distinguir 'A' de 'L'")
plt.show()



# Crear una imagen de 28x28 con todos los píxeles en blanco (valor 0)
image_9 = np.zeros((28, 28))

# Píxeles relevantes (9)
pixels_9 = [(4, 4), (4, 14), (4, 24), (14, 4), (14, 14), (14, 24), (24, 4), (24, 14), (24, 24)]

for pixel in pixels_9:
    image_9[pixel] = 255

# Mostrar la imagen
plt.imshow(image_9, cmap='gray')
plt.title("Píxeles relevantes (9) para distinguir 'A' de 'L'")
plt.show()


# Crear una imagen de 28x28 con todos los píxeles en blanco (valor 0)
image_12 = np.zeros((28, 28))

# Píxeles relevantes (12)
pixels_12 = [
    (2, 4), (2, 14), (2, 24),
    (14, 4), (14, 14), (14, 24),
    (24, 4), (24, 14), (24, 24),
    (18, 4), (18, 14), (18, 24)
]

for pixel in pixels_12:
    image_12[pixel] = 255

# Mostrar la imagen
plt.imshow(image_12, cmap='gray')
plt.title("Píxeles relevantes (12) para distinguir 'A' de 'L'")
plt.show()


# Crear una imagen de 28x28 con todos los píxeles en blanco (valor 0)
image_16 = np.zeros((28, 28))

# Píxeles relevantes (16)
pixels_16 = [
    (2, 4), (2, 14), (2, 24),
    (4, 4), (4, 14), (4, 24),
    (10, 4), (10, 24),
    (14, 4), (14, 14), (14, 24),
    (18, 4), (18, 14), (18, 24),
    (24, 4), (24, 14)
]

for pixel in pixels_16:
    image_16[pixel] = 255

# Mostrar la imagen
plt.imshow(image_16, cmap='gray')
plt.title("Píxeles relevantes (16) para distinguir 'A' de 'L'")
plt.show()




# VISUZALIZACION DE LOS PIXELES CONSIDERADOS POR SOBRE LAS LETRAS 'A' Y 'L' PROMEDIO 

solo_A =data.loc[data[0]=='A']
solo_A = solo_A.iloc[:, 1:]

A = promedio_pixeles(solo_A) 


# Ploteo la imagen transformada
plt.imshow(flip_rotate(A), cmap = 'gray')
plt.title('letra: A')
plt.axis('off')  
plt.show()

flip_rotate(A)


solo_L =data.loc[data[0]=='L']
solo_L = solo_L.iloc[:, 1:]
    
L = promedio_pixeles(solo_L) 


# Ploteo la imagen transformada
plt.imshow(flip_rotate(L), cmap = 'gray')
plt.title('letra: L')
plt.axis('off')  
plt.show()

flip_rotate(L)

#Utilizamos la función "mostrar_superposicion" para poder visualizar los pixeles sobre las letras.
# Mostrar la superposición para la letra 'A'
mostrar_superposicion(flip_rotate(A), image_3, "Letra Promedio: A con Píxeles Relevantes")
mostrar_superposicion(flip_rotate(A), image_3_case2, "Letra Promedio: A con Píxeles Relevantes")
mostrar_superposicion(flip_rotate(A), image_5, "Letra Promedio: A con Píxeles Relevantes")
mostrar_superposicion(flip_rotate(A), image_9, "Letra Promedio: A con Píxeles Relevantes")
mostrar_superposicion(flip_rotate(A), image_12, "Letra Promedio: A con Píxeles Relevantes")
mostrar_superposicion(flip_rotate(A), image_16, "Letra Promedio: A con Píxeles Relevantes")


# Mostrar la superposición para la letra 'L'
mostrar_superposicion(flip_rotate(L), image_3, "Letra Promedio: L con Píxeles Relevantes")
mostrar_superposicion(flip_rotate(L), image_3_case2, "Letra Promedio: L con Píxeles Relevantes") 
mostrar_superposicion(flip_rotate(L), image_5, "Letra Promedio: L con Píxeles Relevantes") 
mostrar_superposicion(flip_rotate(L), image_9, "Letra Promedio: L con Píxeles Relevantes") 
mostrar_superposicion(flip_rotate(L), image_12, "Letra Promedio: L con Píxeles Relevantes") 
mostrar_superposicion(flip_rotate(L), image_16, "Letra Promedio: L con Píxeles Relevantes") 

#%%
# Comparación de modelos

# Definir los datos de exactitud y cantidad de atributos para cada modelo
exactitudes = [0.616, 0.523,0.730,0.734 ,0.731,0.879,0.907]
cantidad_atributos = [3, 3, 5, 9, 6, 12, 16]

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(cantidad_atributos, exactitudes, color='skyblue')
plt.xlabel('Cantidad de Atributos')
plt.ylabel('Exactitud del Modelo')
plt.title('Comparación de Exactitud por Cantidad de Atributos')
plt.xticks(cantidad_atributos)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Conclusión el mejor modelo es el que tiene 16 atributos 


#%%
#
#-------------Tercer Punto--------------------
#
#%%
#Item A.
#Filtro el dataframe, quedándome solo con las vocales.
vocales = data[(data.iloc[:, 0] == 'A') | (data.iloc[:, 0] == 'E') | (data.iloc[:, 0] == 'I') | 
               (data.iloc[:, 0] == 'O') | (data.iloc[:, 0] == 'U')]

X = vocales.iloc[:, 1:]  # Todas las columnas excepto la primera, es decir, los 'pixeles' de cada imagen
y = vocales.iloc[:, 0]   # Solo la primera columna (las vocales)
tipos_vocales=vocales[0].unique()

X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,random_state=1,test_size=0.3) #nos separa un 0.3 (30%) para test

#%%
#Item B 
#Utilizo el modelo de árbol de desición.
#En este caso utilizo diferentes profundidades y criterio entropy.
arbol_decision_1 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 1) #creo el arbol de profundidad 1
arbol_decision_1 = arbol_decision_1.fit(X_dev, y_dev) #lo ajusto a mis datos

plt.figure(figsize= [30,10]) #creo la figura
tree.plot_tree(arbol_decision_1, feature_names = X_dev.columns, class_names = tipos_vocales,filled = True, rounded = True, fontsize = 15)
  
arbol_decision_2 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 2) #creo el arbol de profundidad 2
arbol_decision_2 = arbol_decision_2.fit(X_dev, y_dev) #lo ajusto a mis datos

plt.figure(figsize= [30,10]) #creo la figura
tree.plot_tree(arbol_decision_2, feature_names = X_dev.columns, class_names = tipos_vocales,filled = True, rounded = True, fontsize = 15)
  
arbol_decision_3 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 3) #creo el arbol de profundidad 3
arbol_decision_3 = arbol_decision_3.fit(X_dev, y_dev) #lo ajusto a mis datos

plt.figure(figsize= [30,10]) #creo la figura
tree.plot_tree(arbol_decision_3, feature_names = X_dev.columns, class_names = tipos_vocales,filled = True, rounded = True, fontsize = 15)
  
arbol_decision_5 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 5) #creo el arbol de profundidad 5
arbol_decision_5 = arbol_decision_5.fit(X_dev, y_dev) #lo ajusto a mis datos

plt.figure(figsize= [30,10]) #creo la figura
tree.plot_tree(arbol_decision_5, feature_names = X_dev.columns, class_names = tipos_vocales,filled = True, rounded = True, fontsize = 15)
  

#%%
#Item C 
#Hacemos k-folding para los anteriores arboles.
arboles = []
profundidades = [1, 2, 3, 5]

# Evaluar los modelos utilizando KFold cross-validation
nsplits = 5
kf = KFold(n_splits=nsplits)

resultados = np.zeros((nsplits, len(profundidades)))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, profundidad in enumerate(profundidades):
        arbol = tree.DecisionTreeClassifier(criterion="entropy", max_depth=profundidad)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        # Calcula la matriz de confusión
        tp, tn, fp, fn = matriz_confusion_binaria(kf_y_test.values, pred)
        score = accuracy_score(kf_y_test, pred)  # Usando directamente accuracy_score
        
        resultados[i, j] = score
        
# Calcula el promedio de los scores
scores_promedio = resultados.mean(axis=0)



for i, profundidad in enumerate(profundidades):
    print(f'Score promedio del modelo  con entropy, profundidad {profundidad}: {scores_promedio[i]:.4f}')


#Ahora también hacemos el modelo de árbol de desición para diferentes profundidades pero le cambio el criterio a gini.
arbol_decision_1_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth=1) #profundidad 1
arbol_decision_1_gini.fit(X_dev, y_dev)
plt.figure(figsize=[30, 10])
tree.plot_tree(arbol_decision_1_gini, feature_names=X_dev.columns, class_names=tipos_vocales, filled=True, rounded=True, fontsize=15)
plt.show()

arbol_decision_2_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth=2) #profundidad 2
arbol_decision_2_gini.fit(X_dev, y_dev)
plt.figure(figsize=[30, 10])
tree.plot_tree(arbol_decision_2_gini, feature_names=X_dev.columns, class_names=tipos_vocales, filled=True, rounded=True, fontsize=15)
plt.show()

arbol_decision_3_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)#profundidad 3
arbol_decision_3_gini.fit(X_dev, y_dev)
plt.figure(figsize=[30, 10])
tree.plot_tree(arbol_decision_3_gini, feature_names=X_dev.columns, class_names=tipos_vocales, filled=True, rounded=True, fontsize=15)
plt.show()

arbol_decision_5_gini = tree.DecisionTreeClassifier(criterion="gini", max_depth=5) #profundidad 5
arbol_decision_5_gini.fit(X_dev, y_dev)
plt.figure(figsize=[30, 10])
tree.plot_tree(arbol_decision_5_gini, feature_names=X_dev.columns, class_names=tipos_vocales, filled=True, rounded=True, fontsize=15)
plt.show()

#Hago K-folding con los arboles nuevos.
arboles = []
profundidades_gini = [1, 2, 3, 5]

nsplits = 5
kf = KFold(n_splits=nsplits)

resultados_gini = np.zeros((nsplits, len(profundidades_gini)))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    for j, profundidad in enumerate(profundidades_gini):
        arbol = tree.DecisionTreeClassifier(criterion="gini", max_depth=profundidad)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        # Calcula la matriz de confusión
        tp, tn, fp, fn = matriz_confusion_binaria(kf_y_test.values, pred)
        score_gini = accuracy_score(kf_y_test, pred)  # Usando directamente accuracy_score
        
        resultados_gini[i, j] = score_gini
        
# Calcula el promedio de los scores
scores_promedio_gini = resultados_gini.mean(axis=0)


for i, profundidad in enumerate(profundidades_gini):
    print(f'Score promedio del modelo con Gini, profundidad {profundidad}: {scores_promedio_gini[i]:.4f}')
   
#%%
#Item D
#Creamos un gráfico donde se puede apreciar claramente que el arbol con profundidad 5 y criterio
#entropy es el mejor modelo.
#Gracifo para el criterio "entropy".
plt.subplot(1, 2, 1)
plt.plot(profundidades, scores_promedio, marker='o', color='blue')
for x, y in zip(profundidades, scores_promedio):
    plt.text(x, y, f'{y:.3f}', ha='left', va='bottom')
plt.title('Exactitud promedio (entropy)') 
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Exactitud Promedio')
plt.xticks(profundidades)
plt.grid(True)

# Gráfico para el criterio "gini".
plt.subplot(1, 2, 2)
plt.plot(profundidades_gini, scores_promedio_gini, marker='o', color='green')
for x, y in zip(profundidades_gini, scores_promedio_gini):
    plt.text(x, y, f'{y:.3f}', ha='left', va='bottom')
plt.title('Exactitud promedio (gini)')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Exactitud Promedio')
plt.xticks(profundidades_gini)
plt.grid(True)

plt.tight_layout()
plt.show()

#Una vez obtenido el mejor modelo.
#testeo el modelo con profundidad 5 y criterio entropy
arbol_elegido = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
arbol_elegido.fit(X_dev, y_dev)
y_pred = arbol_elegido.predict(X_dev)

#pruebo el modelo elegido y entrenado en el conjunto dev
tp, tn, fp, fn = matriz_confusion_binaria(y_dev.values, y_pred)
score_arbol_elegido_dev = accuracy_score(y_dev, y_pred)
print('score del arbol elegido con el conjunto de dev:',score_arbol_elegido_dev)

#pruebo el modelo elegido y entrenado en el conjunto eval
y_pred_eval = arbol_elegido.predict(X_eval)       
tp, tn, fp, fn = matriz_confusion_binaria(y_eval.values, y_pred_eval)
score_arbol_elegido_eval = accuracy_score(y_eval, y_pred_eval)
print('score del arbol elegido con el conjunto de eval:',score_arbol_elegido_eval)
    
 
 


