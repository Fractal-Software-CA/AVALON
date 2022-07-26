import pandas as pd
import numpy as np

import findspark
findspark.init()
findspark.find()

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id

import unidecode
import re
import csv

from elasticsearch import Elasticsearch
from time import time
#import import_ipynb
from nltk.tokenize import word_tokenize

from pyspark.sql import Window
import pyspark.sql.functions as f


from Funciones import *


#Método que devuelve un cruce de dataframes por el campo "CIF". Almacena un fichero en formato parquet cuyo nombre es nombre_fichero
def cruza_NIF(solicitudes,entidades,ruta_salida,nombre_fichero,guardarCSV):
    
    df_cruce=solicitudes.alias('a').join(entidades.alias('b'), f.col('a.CIF')==f.col('b.CIF'), how="inner").select('b.Id','a.*')
    df_retorno=df_cruce.dropDuplicates()
    if (guardarCSV):
        df_retorno.toPandas().to_csv(ruta_salida + nombre_fichero + '.csv',
                                     index=False, decimal=',',sep=';',float_format='%.4f',quoting=csv.QUOTE_NONNUMERIC)
    df_retorno.write.mode("overwrite").parquet(ruta_salida + nombre_fichero)
    print(nombre_fichero +' - Registros: ',df_retorno.count())
    return df_retorno

#Método que devuelve los elementos que no cruzan dos dataframes por el campo "CIF". Almacena un fichero en formato parquet cuyo nombre es nombre_fichero
def NO_cruza_NIF(solicitudes,entidades,ruta_salida,nombre_fichero,guardarCSV):
    df_retorno=solicitudes.join(entidades,on="CIF",how="ANTI")
    df_retorno=df_retorno.dropDuplicates()
    if (guardarCSV):
        df_retorno.toPandas().to_csv(ruta_salida + nombre_fichero + '.csv', 
                                     index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    df_retorno.write.mode("overwrite").parquet(ruta_salida + nombre_fichero)
    print(nombre_fichero +' - Registros: ',df_retorno.count())
    return df_retorno

#Obtiene fichero de distancias.
def obtenerDistancias(spark,
                      df_source,df_tarject,
                      nombre_source,nombre_target
                      ,id_source,id_target,
                      Threshold,
                      ruta_salida,Ruta_Input,guardarCSV):
    
    #Leemos un fichero necesario para la funicón de ElasticSearch
    df_municipios = pd.read_csv(Ruta_Input +'MUNICIPIO_PROVINCIA_COMAS.csv').replace(np.nan,'')
    cities = [unidecode.unidecode(i.lower()) for i in list(df_municipios['NOMBRE_PROVINCIA'].unique())]
    all_cities = []
    syn_cities = []
    for i in cities:
        if len(i)>0:
            u = list(set(i.split('/')))
            if len(u)>1:
                syn_cities.append(set(u))
            all_cities+=u
    df_distancias = calcular_Distance_ratcliff_obershelp_ElasticSearch_Ubicacion(spark,
                                                                                             syn_cities,
                                                                                             cities,
                                                                                             df_source,
                                                                                             'Entidad_Norm',
                                                                                             'Provincia_Entidad',
                                                                                             id_source,
                                                                                             None,
                                                                                             None,
                                                                                             df_tarject,
                                                                                             'Entidad_Norm',
                                                                                             'Provincia_Entidad',
                                                                                             id_target,
                                                                                             None,
                                                                                             None,
                                                                                             'indice',
                                                                                             0.93283582,
                                                                                             0.06716418,
                                                                                             None,
                                                                                             Threshold,
                                                                                             10)
   # df_distancias = spark.read.parquet(ruta_salida + "Match_Nombre_distance_Maestro")
    w = Window.partitionBy('source_id')
    df_retorno=df_distancias.withColumn('maxfinal_score', 
                                        f.max('final_score').over(w)).where(
                f.col('final_score')==f.col('maxfinal_score')).drop('maxfinal_score')
    if (guardarCSV):
        df_retorno.dropDuplicates().toPandas().to_csv(ruta_salida + 'Match_Nombre_distance' + '.csv', 
                                                      index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    df_retorno.write.mode("overwrite").parquet(ruta_salida + "Match_Nombre_distance")
    return df_retorno


def NO_cruza_Nombre(df_solicitudes_inicial,df_solicitudes_cruza_nombre,df_distancias,identificador,ruta_salida,nombre_fichero,guardarCVS):
    df_solicitudes_no_cruza_nombre=df_solicitudes_inicial.alias('a').join(
                                                   df_solicitudes_cruza_nombre.alias('b'),
                                                   f.col('a.Identificadores_Origen')==col('b.Identificadores_Origen'),
                                                   how="ANTI").select('a.*')

        
    df_solicitudes_no_cruza_nombre=df_solicitudes_no_cruza_nombre.dropDuplicates()
    
    #ANADIMOS LA COLUMNA DE FINAL_SCORE
    df_solicitudes_no_cruza_nombre.count()
    df_retorno=df_solicitudes_no_cruza_nombre.alias('a').join(df_distancias.alias('b'),
                                                                  f.col('a.'+identificador)==f.col('b.source_id'),
                                                                      how="left"
                                                                ).select(
                                                                'a.*',
                                                                'b.final_score',
                                                                f.col("b.target_names").alias("Entidad_Match"),
                                                                f.col("b.target_municipality").alias("Provincia_Match"))

    df_retorno=df_retorno.dropDuplicates()
    df_retorno.toPandas().to_csv(ruta_salida + nombre_fichero + '.csv', 
                                 index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    df_retorno.write.mode("overwrite").parquet(ruta_salida + nombre_fichero)
    print(nombre_fichero+' - Registros: ',df_retorno.count())
    #creamos df que almacena información de los registros que no han cruzado ni por nif ni por nombre.
    df_retorno=df_retorno.drop('final_score')
    return df_retorno

def cruza_Nombre(df_distancias, df_entrada,identificador, Threshold,ruta_salida,nombre_fichero, nombre_cruce, guardarCVS):
    #Nos quedamos con el mayor score de todos los source_id
    df_distancias_similares=df_distancias.filter(f.col('final_score') >= Threshold)
    df_retorno=df_entrada.alias('a').join(
        df_distancias_similares.alias('b'),
        f.col('a.'+identificador)==f.col('b.source_id'),
        how="inner"
    ).select(
        f.col('b.target_id').alias('Id'),
        'a.*',
        'b.final_score',
        f.col("b.target_names").alias("Entidad_Match"),
        f.col("b.target_municipality").alias("Provincia_Match"),
        f.lit(nombre_cruce).alias("Match")
    )
    #Se quitan duplicados
    df_retorno=df_retorno.dropDuplicates()
    if(guardarCVS):
        df_retorno.toPandas().to_csv(ruta_salida + nombre_fichero + '.csv', 
                                     index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    df_retorno.write.mode("overwrite").parquet(ruta_salida + nombre_fichero)
    print( nombre_fichero +'- Registros: ',df_retorno.count())
    return df_retorno