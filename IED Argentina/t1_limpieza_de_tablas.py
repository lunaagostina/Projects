"""
Estudiantes:Camila Molteni CeccÃ³n
            Luna Praino
            Agustin Quintriqueo
            
Descripción: En este archivo se encontrarán las respectivas limpiezas de las tablas, 
            los códigos para las metricas y luego el guardado de las tablas limpias
            en el directorio Tablas Limpias.
            Para correr las tablas, en "carpeta" se tine que poner la ruta de acceso
            a la carpeta "TablasOriginales". Luego para tablas_limpias hay que colocar
            la ruta de acceso a la carpeta "TablasLimpias"

"""

encoding= 'UTF-8'

import pandas as pd
from inline_sql import sql, sql_val
##---------------

#carpetas para trabajar.
#carpeta corresponde a la ruta de acceso de TablasOriginales.
carpeta = "/Users\\camil\\Documents\\Laboratorio_de_Datos\\TP01-CPQ\\TablasOriginales\\"
#tablas_limpias corresponde a la ruta de acceso de TablasLimpias.
tablas_limpias ="/Users\\camil\\Documents\\Laboratorio_de_Datos\\TP01-CPQ\\TablasLimpias\\"

##---------------
#cargamos las tablas.
flujos_monetarios = pd.read_csv(carpeta+"flujos-monetarios-netos-inversion-extranjera-directa.csv").T.reset_index()
lista_secciones   = pd.read_csv(carpeta+"lista-secciones.csv")
lista_sedes_datos = pd.read_csv(carpeta+"lista-sedes-datos.csv", on_bad_lines='skip')
lista_sedes       = pd.read_csv(carpeta+"lista-sedes.csv")
paises            = pd.read_csv(carpeta+"paises.csv")

##---------------
#Empezamos a limpiar las tablas

#-----FLUJOS MONETARIOS-----
#como calculamos la metrica para la tabla flujos Monetarios. 
def nombreErroneo(df, columna):
    contador = 0
    cantidad_registros = 0
    for valor in df[columna]:
        if valor == 'viet_nam':
            contador += 1 
        elif valor == 'espanna':
            contador += 1 
        elif valor == 'swazilandia':
            contador += 1
        
        cantidad_registros += 1 
    
    metrica = contador/cantidad_registros 
    
    return contador, cantidad_registros, metrica 

#Imprimimos el resultado por pantalla.
print("La metrica con los errores",nombreErroneo(flujos_monetarios, 'index'))


# Modificamos los nombres de las columnas, arreglamos los í­ndices y nos quedamos con las columnas que nos interesan para trabajar.
flujos_monetarios.columns = flujos_monetarios.iloc[0]
flujos_monetarios = flujos_monetarios[1:]
flujos_monetarios.columns = ['paises'] + list(flujos_monetarios.columns[1:])
columna_paises = flujos_monetarios['paises'] #Acá renombramos la columna "index" por "países".
columnas_necesarias = flujos_monetarios.iloc[:, -5:] #Acá nos quedamos con las columnas que vamos a trabajar en un futuro.
flujos_monetarios = pd.concat([columna_paises, columnas_necesarias], axis=1)



#Corregimos los nombres mal escritos en la tabla. Además de que en este momento mejoramos la métrica
flujos_monetarios['paises'].replace({'viet_nam': 'vietnam'}, inplace=True)
flujos_monetarios['paises'].replace({'espanna': 'españa'}, inplace=True)
flujos_monetarios['paises'].replace({'swazilandia': 'suazilandia'}, inplace=True)
#Esos eran los paí­ses mal escritos inclui­dos en las metricas.
 
#Aplicamos la funcion nombreErroneo para ver como afecto a la metrica, la tabla ya modificada 
print("La metrica con los errores corregidos",nombreErroneo(flujos_monetarios,'paises'))


#Estos son paí­ses que modificamos para que a la hora de realizar las consultas sea mas sencillo trabajar con ellos.

flujos_monetarios['paises'].replace({'rep_dem_del_congo': 'republica democratica del congo'}, inplace=True)
flujos_monetarios['paises'].replace({'islas_bermudas': 'islas bermudas'}, inplace=True)
flujos_monetarios['paises'].replace({'bosnia_herzegovina': 'bosnia y herzegoniva'}, inplace=True)
flujos_monetarios['paises'].replace({'botswana': 'botsuana'}, inplace=True)
flujos_monetarios['paises'].replace({'china_rae_de_hong_kong': 'hong kong'}, inplace=True)
flujos_monetarios['paises'].replace({'china_rae_de_macao': 'macao'}, inplace=True)
flujos_monetarios['paises'].replace({'china_provincia_de_taiwan': 'taiwan'}, inplace=True)
flujos_monetarios['paises'].replace({'tfyr_de_macedonia': 'macedonia'}, inplace=True)


#Para flujos_monetarios, vamos a, por cada dato faltante (valor nulo) poner un 0.
flujos_monetarios.fillna(0, inplace=True)


#Nos quedamos con las columnas que nos interensan para trabajar. Y acomodamos la tabla.
flujos_monetarios = pd.melt(flujos_monetarios, id_vars='paises', value_vars=['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01'])
flujos_limpia= sql^ """
                        SELECT paises,variable AS Año, value AS valor
                        FROM flujos_monetarios 

"""
flujos_limpia = sql^"""
                    SELECT REPLACE(paises,'_',' ') AS paises, REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(Año, '2018-01-01', '2018'), '2019-01-01', '2019'), '2020-01-01', '2020'), '2021-01-01', '2021'), '2022-01-01', '2022') AS Año, valor
                    FROM flujos_limpia
        """

#Creamos el archivo csv de la tabla limpia, y lo guardamos en TablasLimpias. 
flujos_limpia.to_csv(tablas_limpias+'flujo_monetariolimpia.csv',index=False)


##---------------
#-----LISTA SECCIONES------

#como calculamos la metrica para la tabla secciones 
def cantidadNulls(df, columna):
    contador = 0
    cantidad_registros = 0
    for valor in df[columna]:
        cantidad_registros += 1 
        if pd.isnull(valor):
            contador += 1 
    
    metrica = contador/cantidad_registros 
    
    return contador, cantidad_registros, metrica 

print("La metrica con los errores",cantidadNulls(lista_secciones, 'correo_electronico') )


#Para lista_secciones, modificamos los valores null, cambiandolos por un 0. Y luego cambiamos los 0 por un - para q quede el mismo tipo de datos
lista_secciones.fillna(0, inplace=True)
lista_secciones.replace('0', '-', inplace=True)


# Aplicamos la funcion cantidadNulls para ver como afecto a la metrica, la tabla ya modificada 
print("La metrica con los errores corregidos",cantidadNulls(lista_secciones,'correo_electronico'))

#Nos quedamos con las columnas que nos interensan para trabajar.
lista_secciones_limpia = sql ^ """
                            SELECT sede_id, sede_desc_castellano AS secciones_desc_castellano, correo_electronico
                            FROM lista_secciones
                          
                        """
                        
#Creamos el archivo csv de la tabla limpia, y lo guardamos en TablasLimpias.
lista_secciones_limpia.to_csv(tablas_limpias+'lista_secciones_limpia.csv',index=False)

##---------------                               

##--------------- 
#------LISTA SEDES DATOS------
#como calculamos la metrica para la tabla sedes datos. Utilizamos la columna "codigo_postal".
def Disponibilidad(df, columna):
    contador = 0
    cantidad_registros = 0
    for valor in df[columna]:
        cantidad_registros += 1 
        
        if pd.isnull(valor):
            contador += 1 
        
        elif valor == '-----':
            contador += 1
        
        elif valor == 'no existe':
            contador += 1 
        
        elif valor == '--------------------':
            contador += 1 
        
        elif valor == 's/c':
            contador += 1 
        
        elif valor == '-':
            contador += 1 
        
        elif valor == '--':
            contador += 1 
        
        elif valor == 'CP':
            contador += 1 
        
        elif valor == '.':
            contador += 1 
    
    metrica = contador/cantidad_registros 
    
    return contador, cantidad_registros, metrica 

print("La metrica con los errores",Disponibilidad(lista_sedes_datos, 'codigo_postal'))


#Para lista_sedes_datos, modificamos los valores null, cambiandolos por un 0. Y luego cambiamos los 0 por un - para q quede el mismo tipo de datos
lista_sedes_datos.fillna(0, inplace=True)
lista_sedes_datos.replace(0, '-', inplace=True)

#Nos quedamos con las columnas que nos interensan para trabajar y lograr la tabla final.
lista_sedes_datos_limpia = sql ^ """
                        SELECT sede_id, sitio_web, region_geografica, pais_iso_2, pais_castellano
                        FROM lista_sedes_datos
                                """
                              
##--------------- 
#-----LISTA SEDES-----
#Hacemos un INNER JOIN entre lista_sedes y lista_sedes_datos_limpias para conseguir la tabla final que necesitamos.
lista_sedes_limpia = sql ^ """
                        SELECT ls.sede_id, ls.sede_desc_castellano, ls.pais_iso_2, ls.pais_castellano,lsd.region_geografica
                        FROM lista_sedes AS ls
                        INNER JOIN lista_sedes_datos_limpia AS lsd
                        ON ls.sede_id = lsd.sede_id
                            """
                            
#Creamos el archivo csv de la tabla limpia, y lo guardamos en TablasLimpias.
lista_sedes_limpia.to_csv(tablas_limpias+'lista_sedes_limpia.csv',index=False)

###_-----------------
#------REDES SOCIALES------
#Primero hicimos una tabla solo con los valores multivaluados de la lista sede datos.
lista=[]
for line in lista_sedes_datos['redes_sociales']:
    filas = line.split(' // ')
    lista.append(filas)
#print(lista)

redes_sociales=pd.DataFrame(data=lista) #Creamos un dataframe con los datos

redes_sociales["id"] = lista_sedes_datos['sede_id'] #Agregamos la columna "id".

redes_sociales = pd.melt(redes_sociales, id_vars='id', value_vars=[0, 1,2,3,4,5,6]) #Acomodamos la tabla como para conseguir que hayan solamente dos columnas.

redes_sociales = redes_sociales.drop(['variable'],axis=1)

redes_sociales = redes_sociales[redes_sociales['value']!='-']
redes_sociales = redes_sociales[redes_sociales['value']!=' ']
redes_sociales = redes_sociales[redes_sociales['value'].notnull()]

#Finalmente, terminamos de lograr la tabla con SQL y le cambiamos el nombre de la columna "value" por "red_social" ya que nos parecia mas descriptivo.
redes_sociales= sql^"""
                SELECT id,value AS red_social
                FROM redes_sociales
"""

#Creamos el archivo csv de la tabla limpia, y lo guardamos en TablasLimpias. 
redes_sociales.to_csv(tablas_limpias+'redes_sociales.csv',index=False)

##--------------- 
#------PAISES------
#Para lista_sedes_datos, modificamos los valores null, cambiandolos por un 0. Y luego cambiamos los 0 por un - para q quede el mismo tipo de dato.
paises.fillna(0, inplace=True)
paises.replace(0, '-', inplace=True)


#Nos quedamos con las columnas que vamos a usar, pusimos todas las letras mayusculas en minusculas y sacamos acentos.
paises_tabla_limpia = sql ^ """
                SELECT REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(nombre),'á','a'),'é','e'),'í','i'),'ó','o'),'ú','u'),'-',' ') AS nombre, " iso2" AS iso2
                FROM paises
        
                     """
#Cambiamos algunos nombres de países para mayor practicidad.
paises_tabla_limpia['nombre'].replace({'macedônia': 'macedonia'}, inplace=True)  
paises_tabla_limpia['nombre'].replace({'estados unidos de america': 'estados unidos'}, inplace=True) 
                  
#Creamos el archivo csv de la tabla limpia, y lo guardamos en TablasLimpias. 
paises_tabla_limpia.to_csv(tablas_limpias+'paises_limpia.csv',index=False)



