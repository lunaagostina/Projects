import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns

def calcularLU(A):
    """
    Recibe: una matriz A 
    Devuelve: las matrices L (triangular inferior), U (triangular superior) y P(matriz de permutación)
    correspondientes a la factorización PA = LU
    """
    m = A.shape[0]  # cantidad de filas
    n = A.shape[1]  # cantidad de columnas
    Ac = A.copy()   # copia de la matriz original

    P = np.eye(m)   # matriz de permutación
    L = np.eye(m)   # matriz triangular inferior inicializada como la identidad
    U = Ac.copy()   # copia de la matriz original para U

    #En primer lugar, nos fijamos si la matriz es cuadrada. Es decir, si tiene la misma cantidad de filas que de columnas

    if m != n:
        print('Matriz no cuadrada')
        return

    #Si es cuadrada, seguimos con lo siguiente:
        
        #Armamos un for que tome los índices de la matriz (0 <= k < m)

    for k in range(m):
        if U[k, k] == 0:
            #Si los elementos de las diagonales son = 0, vuelvo a armar un for que avance a la siguiente fila
            #pues en la diagonal quiero elementos != 0
          for j in range(k + 1, m):
                if U[j, k] != 0:
                    #Si el elemento donde estamos parados es != 0, intercambio toda la fila de la matriz U con la que tenía anteriormente
                    #de manera tal que en la diagonal tengo elementos != 0
                    #Análogamente para P y L
                    
                    U[[k, j], :] = U[[j, k], :]
                    P[[k, j], :] = P[[j, k], :]
                    L[[k, j], :k] = L[[j, k], :k]
                    break

        # Eliminar los elementos por debajo de la diagonal
        for i in range(k + 1, m):
            if U[i, k] != 0:
                #Comienzo a escalonar, si el elemento U[i][k] es distinto de 0, armo el coeficiente de triangulación
                #que luego voy a ubicar en la posición que corresponda de la matriz L.
                #Por último, a la fila i de la matriz U le resto coef * la fila k de U
                coef = U[i, k] / U[k, k]
                L[i, k] = coef
                U[i, :] = U[i, :] - coef * U[k, :]

    return L, U, P

def inversaLU(L, U, P=None):
    """
    Recibe las matrices L, U y P (en caso de que exista)
    Devuelve: la inversa de la matriz original, calculada a partir de las 3 anteriores.
    Ahora se usa solve_triangular con vectores columna.
    """
    m = L.shape[0]
    Id = np.eye(m)  # matriz identidad
    Inv = np.zeros_like(Id)  # matriz para almacenar la inversa

    # Resolver L * Y = P * I para cada columna de la identidad
    for i in range(m):
        e_i = Id[:, i]  # obtener la i-ésima columna de la identidad, o bien el canónico e_i
        y_i = scipy.linalg.solve_triangular(L, np.dot(P, e_i), lower=True) #Notar que np.dot(P, e_i) es un vector
        
        # Resolver U * x_i = y_i para encontrar cada columna de la inversa
        Inv[:, i] = scipy.linalg.solve_triangular(U, y_i, lower=False)

    return Inv

def obtenerInversa(A):
    """
    Recibe: Matriz cuadrada inversible
    Devuelve: Inversa de la matriz mediante decomposición LU
    """
    L,U,P = calcularLU(A)
    return inversaLU(L,U,P)

def generar_matriz_random_no_singular(size):
    """  
    Recibe: El tamaño deseado de la matriz (int)
    Devuelve: Matriz (shape (size,size)) con números random, no singular 
    """
    while True:
        A = np.random.rand(size, size)
        A = np.eye(size)-A
        if np.linalg.det(A) != 0:
            return A

def obtener_matriz_insumo_producto(df,PAIS_fila,PAIS_columna):
    """
    Recibe un dataframe con matrices insumo producto, y dos claves (strings) correspondientes a dos países
    Devuelve: la matriz insumo producto intra (si PAIS_fila y PAIS_columna son iguales) o interregional (caso contrario).
    """
    columnas_objetivo = [f'{PAIS_columna}s{i}' for i in range(1, 41)]  # Lista de columnas desde 'PAIS_columna's1 a 'PAIS_columna's40
    df_filtrada = df[columnas_objetivo]
    df_filtrada = df_filtrada[df.iloc[:, 0] == PAIS_fila] # Filtra ahora por PAIS_fila
    return df_filtrada.to_numpy()

def obtener_vector_prod_total(df,PAIS_fila):
    """
    Recibe un dataframe con matrices insumo producto y devuelve el vector producción del país 
    cuya clave es indicada en PAIS_fila
    """
    df_prod_total = df['Output'] # Toma columna de producción total
    df_prod_total = df_prod_total[df.iloc[:, 0] == PAIS_fila] # Filtra ahora por PAIS_fila
    return df_prod_total.to_numpy()

def visualizar_mapa_de_calor(array,titulo,subtitulo):
    """
    Plotea el contenido de una matriz en modo de mapa de calor, en escala logarítmica
    """
    plt.figure(figsize=(4, 3))
    array_log = np.log10(array + 1e-10) #Evita evaluar log10 de 0
    plt.imshow(array_log, cmap='viridis')
    colorbar = plt.colorbar()
    colorbar.set_label('Valores (Escala logarítmica)', fontsize=11)
    plt.title(titulo+'\n'+subtitulo, fontsize=12)
    plt.show()
    return

def reemplazar_por_1_filas_nulas(array):
    """
    Recibe:
        Un array que puede ser uni- o bi-dimensional
    Devuelve:
        El mismo array pero en todas las filas con únicamente elementos nulos, reemplazados por 1
    """
    if len(array.shape) == 1: # Vector
        array[array == 0] = 1
    else: # Matriz
        for i in range(array.shape[0]):
            if np.all(array[i, :] == 0):  # Chequea si todos los elementos de la matriz son nulos
                array[i, :] = 1  # En caso afirmativo, reemplaza por 1
    return array

def obtener_delta_p_modelo_simple(Arr,delta_dr):
    """
    Recibe:
        Arr (shape (n,n)): matriz de coeficientes técnicos de la economía intraregional de la región r
        delta_dr (shape (n,)) : variación en la demanda de la región r
    Devuelve:
        variación en la producción evaluando el modelo simple (shape (n,))
    """
    Irr = np.eye(Arr.shape[0])
    Inversa_Leontief = Irr-Arr
    Leontief = obtenerInversa(Inversa_Leontief)
    return Leontief@delta_dr

def obtener_delta_p_formula_completa(Arr,Ass,Ars,Asr,delta_dr):
    """
    Recibe:
        Arr (shape (n,n)): matriz de coeficientes técnicos de la economía intraregional de la región r
        Ass (shape (m,m)): matriz de coeficientes técnicos de la economía intraregional de la región s
        Ars (shape (n,m)): idem anterior pero inter-regionales
        Asr (shape (m,n)): idem anterior pero inter-regionales
        delta_dr (shape (n,)) : variación en la demanda de la región r
    Devuelve:
        variación en la producción evaluando la fórmula completa (shape (n,))
    """
    Irr = np.eye(Arr.shape[0])
    Iss = np.eye(Ass.shape[0])
    Inversa_Leontief_ss = Iss-Ass
    Leontief_ss = obtenerInversa(Inversa_Leontief_ss)
    Inversa_Leontief_modificado = Irr - Arr - Ars@(Leontief_ss@Asr)
    Leontief_modificado = obtenerInversa(Inversa_Leontief_modificado)
    return Leontief_modificado@delta_dr

def obtener_demanda_sector_r_formula_completa(Arr,Ars,pr,ps):
    """
    Recibe:
        Arr (shape (n,n)): matriz de coeficientes técnicos de la economía intraregional de la región r
        Ars (shape (n,m)): idem anterior pero inter-regionale con s
        pr (shape (n,)) : producción de la región r
        ps (shape (m,)) : producción de la región s

    Devuelve:
        demanda del sector r evaluando la fórmula completa
    """
    Irr = np.eye(Arr.shape[0])
    Inversa_Leontief = Irr-Arr
    return Inversa_Leontief@pr - Ars@ps

def obtener_demanda_sector_r_modelo_simple(Arr,pr):
    """
    Recibe:
        Arr (shape (n,n)): matriz de coeficientes técnicos de la economía intraregional de la región r
        pr (shape (n,)) : producción de la región r

    Devuelve:
        demanda del sector r evaluando el modelo simple
    """
    Irr = np.eye(Arr.shape[0])
    Inversa_Leontief = Irr-Arr
    return Inversa_Leontief@pr

def visualizar_cambios_en_produccion(delta_p, titulo, subtitulo):
    """
    Recibe:
        delta_p (shape (n,)): variación de producción en un dado sector, en millones de dólares

    Devuelve:
        Nada. Solo muestra en pantalla un gráfico de barras.
    """
    plt.figure(figsize=(20, 3))
    sectores = [f's{i}' for i in range(1, len(delta_p)+1)] #Genera labels para cada sector
    ax = sns.barplot(x=sectores, y=delta_p)
    plt.xlabel('Sectores')
    plt.ylabel('Millones de dólares')
    plt.title(titulo+'\n'+subtitulo, fontsize=12)
    plt.show()