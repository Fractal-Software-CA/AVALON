import pyspark
import pandas as pd
import numpy as np

from pyspark.sql.types import *
from pyspark.sql.functions import *


from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.sql import functions as F

from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from pyspark.sql import *
from pyspark import StorageLevel
from pyspark.sql.functions import udf

from pyspark.ml.feature import StopWordsRemover

import copy

from pyspark.sql.functions import pandas_udf

import textdistance

from tqdm import tqdm

#from past.builtins import xrange

from pyspark.sql.types import *

import pyspark.sql.types as T

import unidecode
import re
import csv
import datetime
from nltk.tokenize import word_tokenize

from elasticsearch import Elasticsearch

import nltk




def format_results(results,df1columns,df2columns):
    names_source = []
    names_target = []
    columnsSourcesNames1 = [[]]*len(df1columns)
    columnsTargetNames1 = [[]]*len(df2columns)
    for i in tqdm(results):

        source_name = results[i]['source_name']
        columnsSourcesNamesResults = list(map(lambda x: results[i][df1columns[x]], range(len(df1columns))))
        for element in results[i]['matching_names_by_elasticsearch']:
            target_name = element[1]
            names_source.append(source_name)
            names_target.append(target_name)
            columnsSourcesNames1 = list(map(lambda k: columnsSourcesNames1[k] + [columnsSourcesNamesResults[k]], range(len(df1columns))))
            columnsTargetNamesResults = list(map(lambda x: element[x+2], range(len(df2columns))))
            columnsTargetNames1 = list(map(lambda x: columnsTargetNames1[x] + [columnsTargetNamesResults[x]], range(len(df2columns))))    

    dicCol = dict([(df1columns[l], columnsSourcesNames1[l]) for l in range(len(df1columns))])
   
    data = {'source_name':names_source, 'target_name':names_target}

    data.update(dicCol)

    dicCol = dict([(df2columns[l], columnsTargetNames1[l]) for l in range(len(df2columns))])
    
    data.update(dicCol)

    result = pd.DataFrame(data)
            
    return result




def Distance_ratcliff_obershelp(column1, column2):
    return textdistance.ratcliff_obershelp(column1, column2)

udf_Distance_ratcliff_obershelp = F.udf(Distance_ratcliff_obershelp, FloatType())





def index_es(df,target_name_column,index_name):
    '''
    inputs:
        df : pd.DataFrame object containing the two columns source names and target names
        target_names : type: str, represents the the target names column
        index_name : type: str, represents the index of elastic search object
    output:
        return elasticsearch object indexed, and initialized with target names
    
    '''

    target_names = [name.lower() for name in list(df[target_name_column])]
    
    try:
        es = Elasticsearch(sniff_on_connection_fail=True,
            sniff_on_start=True, min_delay_between_sniffing=600,
            request_timeout=600, sniff_timeout=300,
            max_retries=5, retry_on_timeout=True)
        for i in list(es.indices.get_alias("*").items()):
            es.indices.delete(index=list(i)[0])
    except:
        es = Elasticsearch('http://localhost:9200/', sniff_on_connection_fail=True,
            sniff_on_start=True, min_delay_between_sniffing=600,
            request_timeout=600, sniff_timeout=300,
            max_retries=5, retry_on_timeout=True)
        for i in list(es.indices.get_alias().items()):
            es.indices.delete(index=list(i)[0])
    
    es.indices.delete(index='test-index', ignore=[400, 404])
    
    for name in tqdm(target_names):
        i= target_names.index(name)
        doc = {'entity':name}
        es.index(index=index_name, id=i, document=doc)  
    return es




def get_matching_entities_by_elasticSearch(df1,
                                           source_name_column,
                                           df1columns,
                                           df2,
                                           target_name_column,
                                           df2columns,
                                           index_name,
                                           n):
    '''
    inputs:
        df : pd.DataFrame object containing the two columns source names and target names
        source_names : type: str, represents the the source names column
        index_name : type: str, represents the index of elastic search object
        size: type: int , number of targets to extract by elasticseasource_pd_altas_nuevas['source_municipality'] = source_pd_altas_nuevas['source_municipality'].astype('Int64')rch, default is 20,
                max for this paremeter is 10 000 
    output:
        return dict object results
    '''
    
    df1 = df1.toPandas()
    df1.replace(to_replace=[None], value='', inplace=True)
    df2 = df2.toPandas()
    df2.replace(to_replace=[None], value='', inplace=True)
    es = index_es(df2,target_name_column,index_name)

    source_name = list(df1[source_name_column])
    
    columnsSourcesNames = list(map(lambda x: list(df1[df1columns[x]]), range(len(df1columns))))
    # columnsSourcesNames = [[]]*len(df1columns)
    # for k in range(len(df1columns)):
    #     columnsSourcesNames[k] = list(df1[df1columns[k]])
    
    columnstargetNames = list(map(lambda x: list(df2[df2columns[x]]), range(len(df2columns))))
    # columnstargetNames = [[]]*len(df2columns)
    # for k in range(len(df2columns)):
    #     columnstargetNames[k] = list(df2[df2columns[k]])
    
    source_idx=0
    results = {}

    for entity_i in tqdm(source_name):

        matching_entities = []
        query = {"from": 0,
                 "size": n,
                 "query": {"match": {"entity":{'query':entity_i,'fuzziness':'AUTO'}}},
                 "sort":[{'_score':{'order':'desc'}}]}
        resp = es.search(index=index_name, body=query)
        
        
        for element in resp['hits']['hits']:

            target_idx = int(element['_id'])
            entity_j = element['_source']['entity']
            
            firstMatch = [target_idx,entity_j]
            
            secondMatch = [columnstargetNames[l][target_idx] for l in range(len(df2columns))]
#             secondMatch = []
            
#             for l in range(len(df2columns)):
#                 secondMatch = secondMatch + [columnstargetNames[l][target_idx]]
            
            match = tuple(firstMatch + secondMatch)
            matching_entities.append(match)
        
        dicCol = dict([(df1columns[l], columnsSourcesNames[l][source_idx]) for l in range(len(df1columns))])
#         dicCol = {}
        
#         for l in range(len(df1columns)):
#             dicCol[df1columns[l]] = columnsSourcesNames[l][source_idx]
        

        results[source_idx] = {'source_name':entity_i, 'matching_names_by_elasticsearch':matching_entities}
        results[source_idx].update(dicCol)
        source_idx +=1
    results = format_results(results, df1columns,df2columns)    
    return results



def get_matching_by_elasticSearch_Distance_ratcliff_obershelp(spark,results):
    

    results = spark.createDataFrame(results)
    stopwords_es = nltk.corpus.stopwords.words('spanish')
    results = results.withColumn("aux", F.split("source_name", "\\s+"))
    remover = StopWordsRemover(stopWords=stopwords_es, inputCol="aux", outputCol="source_names_stopwords")
    results = remover.transform(results).withColumn("source_names_stopwords", F.array_join("source_names_stopwords", " "))
    results = results.withColumn("aux", F.split("target_name", "\\s+"))
    remover = StopWordsRemover(stopWords=stopwords_es, inputCol="aux", outputCol="target_names_stopwords")
    results = remover.transform(results).withColumn("target_names_stopwords", F.array_join("target_names_stopwords", " "))
    
    return results.withColumn('final_score', udf_Distance_ratcliff_obershelp(F.col('source_names_stopwords'), F.col('target_names_stopwords'))).drop('aux').withColumn('final_score', F.when(F.col('source_names_stopwords') != (F.col('target_names_stopwords')), F.col('final_score')).otherwise(F.lit(1)))

# Funcion que aplica las tecnicas de limpieza
def normalizarTexto(texto):
    """
    Ejemplo de uso:
    -----------------
    df_final = df.withColumn("NombreColOutput",UDF_normalizarTexto("NombreCol2Normalizar"))
    """
    try:
        # 1.1. Eliminar tildes
        texto=unidecode.unidecode(texto)
        # 1.2. Elimina caracteres especiales y digitos lo reemplaza por un espacio en Blanco
        texto=re.sub("(\\W)+"," ",texto)
        # 1.3. Eliminar la concatenación de espacios en blanco (quedarnos únicamente con un espacio en blanco entre palabras)
        # 1.4. Pasar a minúsculas:
        texto = texto.lower()

        texto=" ".join(texto.split())

        return texto
    except:
        return None

#Transformar funcion a UDF para usar pyspark
UDF_normalizarTexto=udf(normalizarTexto, StringType())


def save_csv_parquet(var, Flag_csv, ruta):
    if Flag_csv == True:
        var.toPandas().to_csv(f'{ruta}.csv', decimal=",", index=False )
        
        
        var.toPandas().to_parquet(f'{ruta}.parquet',  index=False)
    else:
        var.toPandas().to_parquet(f'{ruta}.parquet',  index=False)









