
"""
Estudiantes:Camila Molteni Ceccón
            Luna Praino
            Agustin Quintriqueo
            
Descripción: En este archivo, se encontrarán los reportes de las consultas SQL. 
            Además se podran visulizar las herramientas de visulización (gráficos) 
            correspondientes a la informacion requerida. 
            Para correr las tablas  primero se tiene que colocar la ruta de acceso
            a la carpeta de "TablasLimpias".
"""
encoding= 'UTF-8'
import pandas as pd
from inline_sql import sql, sql_val 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from   matplotlib import ticker   
import seaborn as sns 
#------------------
#carpetas para trabajar.
#carpeta correspodiente a la ruta de acceso del directorio TablasLimpias.
tablas_limpias ="/Users\\camil\\Documents\\Laboratorio_de_Datos\\TP01-CPQ\\TablasLimpias\\"
#carpeta correspodiente a la ruta de acceso del directorio Anexo.
anexo="/Users\\camil\\Documents\\Laboratorio_de_Datos\\TP01-CPQ\\Anexo\\"
##---------------
#cargamos las tablas
flujos_monetarios = pd.read_csv(tablas_limpias+"flujo_monetariolimpia.csv")
lista_secciones   = pd.read_csv(tablas_limpias+"lista_secciones_limpia.csv")
lista_sedes       = pd.read_csv(tablas_limpias+"lista_sedes_limpia.csv")
paises            = pd.read_csv(tablas_limpias+"paises_limpia.csv")
redes_sociales    = pd.read_csv(tablas_limpias+"redes_sociales.csv")
##---------------
##CONSULTAS SQL
#ITEM 1 
#Primero hacemos una cosulta para quedarnos con los paises y su valor de flujo monetario en el 2022.
flujo = sql^"""
                SELECT paises,Año,valor
                FROM flujos_monetarios
                WHERE Año = 2022
                
"""
#Segundo hacemos una consulta para saber cuantas secciones hay por sede.
cant_secciones = sql^"""
                            SELECT DISTINCT ls.sede_id,COUNT(secciones_desc_castellano) AS cantidad_secciones 
                            FROM lista_secciones AS ls
                            GROUP BY ls.sede_id
                    """
#Tercero hacemos una consulta para saber la cantidad de sedes que hay por país.
cant_sedes = sql^"""
                    SELECT  p.nombre,COUNT(ls.sede_id) AS cantidad_sedes,ANY_VALUE(ls.sede_id) AS sede_id
                    FROM lista_sedes AS ls
                    INNER JOIN paises as p
                    ON ls.pais_iso_2 = p.iso2
                    GROUP BY p.nombre
"""     
#Por último, unimos todo y realizamos la consulta final.              
consulta_1=sql^"""
                SELECT cs.nombre AS pais,cs.cantidad_sedes AS sedes, ROUND((csec.cantidad_secciones/cs.cantidad_sedes),2) AS secciones_promedio, ROUND(flujo.valor,2) AS IED_2022
                FROM cant_sedes AS cs
                INNER JOIN cant_secciones AS csec
                ON cs.sede_id = csec.sede_id
                INNER JOIN flujo
                ON cs.nombre = flujo.paises
                ORDER BY cs.cantidad_sedes DESC
"""

#Creo el archivo csv de la consulta. El mismo se encuentra en la carpeta Anexo para una mejor visualización. 
consulta_1.to_csv(anexo+'consulta_1.csv',index=False)

##---------------
#ITEM 2
#Armamos una tabla con las regiones y la cantidad de países con sedes argentinas por región (enunciado a)
consultaSQL = """
                SELECT region_geografica, COUNT(region_geografica) AS Paises_con_sedes_argentinas
                FROM lista_sedes_limpia
                GROUP BY region_geografica
                ORDER BY region_geografica
"""

consulta_2_a = sql ^ consultaSQL

#-----------
#Hacemos un INNER JOIN entre lista_sedes_limpia y nombre, para que nos devuelva
#con la región correspondiente a cada país.

region_y_paises =  sql ^ """
                SELECT region_geografica, nombre
                FROM lista_sedes_limpia
                INNER JOIN paises
                ON paises.iso2 = lista_sedes_limpia.pais_iso_2
                GROUP BY region_geografica, nombre
                ORDER BY region_geografica, nombre
"""

#-----------
#Hacemos un INNER JOIN entre la tabla anterior y flujo_monetario_limpia, para
#obtener 'flujo_monetario_con_regiones', donde cada país, además de tener su región correspondiente, tiene
#el valor de flujo respectivo a cada año.

flujo_monetario_con_regiones = sql ^"""
                SELECT nombre, region_geografica, Año, valor
                FROM region_y_paises
                INNER JOIN flujos_monetarios
                ON nombre = paises
                GROUP BY nombre, region_geografica, Año, valor
                ORDER BY nombre, region_geografica, Año, valor
"""

#----------

#Finalmente, calculamos el promedio de IED de cada región respecto al año 2022.

consultaSQL = """
            SELECT consulta_2_a.region_geografica, Paises_con_sedes_argentinas, AVG(valor) AS Promedio_IED_2022
            FROM consulta_2_a
            INNER JOIN flujo_monetario_con_regiones AS fmr
            ON consulta_2_a.region_geografica = fmr.region_geografica AND Año = 2022
            GROUP BY consulta_2_a.region_geografica, Paises_con_sedes_argentinas
            ORDER BY Promedio_IED_2022 DESC
"""

consulta_2_b = sql ^ consultaSQL

#----------
#ITEM 3
#Hacemos un INNER JOIN entre lista_Sedes y redes_sociales, tales que sede_id = id.

paises_id = sql ^"""
            SELECT nombre, sede_id
            FROM paises
            INNER JOIN lista_sedes_limpia
            ON lista_sedes_limpia.pais_iso_2 = paises.iso2
            GROUP BY nombre, sede_id
            ORDER BY nombre, sede_id
"""

#---------
#Hacemos un join entre paises_id y redes_sociales y nos quedamos con lo que pide el enunciado.
paises_con_redes = sql ^"""
        SELECT nombre,
            CASE
                WHEN red_social LIKE '%twitter%' THEN 'Twitter'
                WHEN red_social LIKE '%facebook%' THEN 'Facebook'
                WHEN red_social LIKE '%instagram%' THEN 'Instagram'
                ELSE red_social
            END AS Red_social
        FROM redes_sociales
        INNER JOIN paises_id
        ON paises_id.sede_id = redes_sociales.id
        GROUP BY nombre, Red_social
        ORDER BY nombre
"""

consulta_3 = sql ^"""
        SELECT nombre, COUNT(Red_social) AS Total
        FROM paises_con_redes
        GROUP BY Nombre
        
"""

#Creo el archivo csv de la consulta. El mismo se encuentra en la carpeta Anexo para una mejor visualización. 
consulta_3.to_csv(anexo+'consulta_3.csv',index=False)

#----------
#ITEM 4
#Hacemos un join entre lista_sedes_limpia y paises_limpia, tales que sus iso2 
#coinciden, para quedarnos con el nombre de los países y el id de la sede.


paises_id = sql ^"""
            SELECT nombre, sede_id
            FROM paises
            INNER JOIN lista_sedes_limpia
            ON lista_sedes_limpia.pais_iso_2 = paises.iso2
            GROUP BY nombre, sede_id
            ORDER BY nombre, sede_id
"""

#---------
#Hacemos un join entre paises_id y redes_sociales y nos quedamos con lo que pide el enunciado
#Tenemos en cuenta twitter, facebook, instagram y youtube, si hay una red social diferente o alguna
#descripta con @, el url y el value quedan igual.

consulta_4 = sql ^"""
        SELECT nombre AS pais, paises_id.sede_id, red_social AS URL, 
            CASE
                WHEN red_social LIKE '%twitter%' THEN 'Twitter'
                WHEN red_social LIKE '%facebook%' THEN 'Facebook'
                WHEN red_social LIKE '%instagram%' THEN 'Instagram'
                WHEN red_social LIKE '%youtube%' THEN 'Youtube'
                ELSE red_social
            END AS Red_social
        FROM redes_sociales
        INNER JOIN paises_id
        ON paises_id.sede_id = redes_sociales.id
        GROUP BY nombre, sede_id, URL, Red_social
        ORDER BY nombre ASC, sede_id ASC, Red_social ASC, URL asc
"""
#Creo el archivo csv de la consulta. El mismo se encuentra en la carpeta Anexo para una mejor visualización.
consulta_4.to_csv(anexo+'consulta_4.csv',index=False)

##---------------------------------------------------------------------------
#VISUALIZACION
#PUNTO 1
#Primero hacemos una consulta como para quedarnos con lo pedido.
cant_sed = sql^"""
                SELECT region_geografica, COUNT(sede_id) AS cantidad
                FROM lista_sedes
                GROUP BY region_geografica
                ORDER BY cantidad DESC
"""
#Generamos el grafico de barras, que es lo que nos pareció mas conveniente para el punto.Y lo mejoramos visualmente.
fig, ax = plt.subplots()

ax.bar(data=cant_sed, x='region_geografica', height='cantidad',color='pink',edgecolor='red',linewidth = 1.5,width = 0.9,align='center')

plt.rcParams['font.family'] = 'sans-serif'
ax.set_title('Cantidad de sedes por región geografica')
ax.set_xlabel('Región Geografica', fontsize='large')                       
ax.set_ylabel('Cantidad', fontsize='medium')
ax.tick_params(axis='x', labelrotation=80)

ax.set_xticks(range(0,10,1))
ax.set_yticks([])                          
ax.bar_label(ax.containers[0], fontsize=11,color= 'red')
ax.set_ylim(0,50)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,}")); #colocamos en separador de miles.
#---------
#PUNTO 2
#Hacemos una tabla con los promedios de cada región por año. Aprovechamos los df creados en los ejercicios de sql y los modificamos a nuestra convenciencia.

consultaSQL = """
            SELECT consulta_2_a.region_geografica, AVG(valor) AS Promedio_IED, Año
            FROM consulta_2_a
            INNER JOIN flujo_monetario_con_regiones AS fmr
            ON consulta_2_a.region_geografica = fmr.region_geografica
            GROUP BY consulta_2_a.region_geografica, Paises_con_sedes_argentinas, Año
            ORDER BY consulta_2_a.region_geografica, Paises_con_sedes_argentinas, Año
            
"""

flujo_con_promedios = sql ^ consultaSQL

mediana = sql ^ """
            SELECT region_geografica, median(Promedio_IED) AS mediana
            FROM flujo_con_promedios
            GROUP BY region_geografica
            ORDER BY mediana DESC
"""


fig, ax = plt.subplots()

#ax.boxplot(flujo_con_promedios['Promedio_IED'], showmeans = True)

ax = sns.boxplot(x = 'region_geografica',
                 y = 'Promedio_IED',
                 data = flujo_con_promedios,order = ['AMÉRICA  DEL  NORTE', 'ASIA', 'OCEANÍA', 'AMÉRICA  DEL  SUR', 'EUROPA  OCCIDENTAL',
                                                     'EUROPA  CENTRAL  Y  ORIENTAL', 'ÁFRICA  DEL  NORTE  Y  CERCANO  ORIENTE',
                                                     'AMÉRICA  CENTRAL  Y  CARIBE', 'ÁFRICA  SUBSAHARIANA'],color = 'pink', meanprops={"marker": "*",
                       "markerfacecolor": "black", "markeredgecolor" : "black",
                       "markersize": "10"}, showmeans = True, ax = ax)


ax.tick_params(axis='x', labelrotation=80)

ax.set_title('Promedios de IED por región geográfica')
ax.set_xlabel('Región geográfica')
ax.set_ylabel('Promedio del IED')
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,}"));#acá utilizo el separador de miles
#ax.legend(title = 'Región geográfica')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

#ax.set_xticklabels(['EUROPA  OCCIDENTAL', 'ÁFRICA  SUBSAHARIANA', 'ASIA', 'ÁFRICA  DEL  NORTE  Y  CERCANO  ORIENTE',
#                    'AMÉRICA  DEL  SUR', 'OCEANÍA', 'AMÉRICA  CENTRAL  Y  CARIBE', 'AMÉRICA  DEL  NORTE', 'EUROPA  CENTRAL  Y  ORIENTAL'])

plt.show()


#---------
#PUNTO 3 
#Hacemos un SCATTER PLOT porque al tener un rango de valores tan amplio nos pareció lo mas conveniente. Y lo mejoramos visualmente.
fig, ax = plt.subplots(figsize = (15, 9))
plt.scatter(data = consulta_1, x='pais', y='IED_2022',s=60,color='#D5476D')
plt.rcParams['font.family'] = 'sans-serif'
ax.set_title('Paises y IED-2022',fontsize='large') # Titulo del gráfico
ax.set_xlabel('Países', fontsize='large')   # Nombre eje X           
ax.set_ylabel('IED 2022',  fontsize='large')
ax.tick_params(axis='x', labelrotation=85)
ax.tick_params(axis='x', labelsize=13)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,}")); #acá utilizo el separador de miles
ax.set_xticks(range(-1,63,1))
ax.set_xlim(-1,64)
plt.show()





