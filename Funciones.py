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


# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df, spark):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return spark.createDataFrame(pandas_df, p_schema)



def Remove_duplicate(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

# Funcion que aplica las tecnicas de limpieza
def EliminarEspaciosExterioresTexto(texto):
    #Eliminamos espacios en el texto al inicio y al final de la cadena.
    #texto=" ".join(re.split(r"\s+", texto))
    return " ".join(str(texto).split())

#Transformar funcion a UDF para usar pyspark
UDF_EliminarEspaciosExterioresTexto = udf(EliminarEspaciosExterioresTexto, StringType())


#https://mail.python.org/pipermail/python-es/2009-April/024466.html
def pre_process_normalize(text):
    
    text=unidecode.unidecode(text)
    
    # Lowercase
    text=text.lower()
    
    # Remove special characters
    text=re.sub("(\\W)+"," ",text) #mantener digitos
    #text=re.sub("(\\d|\\W)+"," ",text) #eliminar digitos
    
    text=re.sub("\(.*\)|\s-\s.*", "", text) #version anterior
    #text=re.sub("\(.*\)|\s*\w\s*", "", text)
    
    # Convert to list from string
    text = text.split()
    
    # Remove stopwords
    #text = [word for word in text if word not in stop_words]
    
    text = ' '.join(text)
    
    try:
        int(text)
        text = ''
    except:
        pass
    
    return text



def Obtener_Fisica_Juridica(valor):
    CLAVES_NIF = 'XYZ'
    if valor == None:
        return 'No CIF'
    if valor[0] in CLAVES_NIF or valor[0].isdigit():
        return 'F'
    else:
        return 'J'

UDF_Obtener_Fisica_Juridica = udf(Obtener_Fisica_Juridica, StringType())



def validarNIF(valor):
    TABLA_NIF='TRWAGMYFPDXBNJZSQVHLCKE'       # Valores para validar el NIF
    
    CLAVES_NIF1 = 'LKM'                       # Son especiales, se validan
                                              # como CIFs
    CLAVES_NIF2 = 'XYZ'
    CLAVES_NIF = CLAVES_NIF1 + CLAVES_NIF2


    """
    Nos indica si un NIF es valido.
    El valor debe estar normalizado
    @note:
    - ante cualquier problema se valida como False
    """
    bRet = False
    if len(valor) == 9:
        try:
            if valor[0] in CLAVES_NIF1:
                bRet = validarCIF(valor)
            else:
                num=None
                if valor[0] in CLAVES_NIF2:
                    pos = CLAVES_NIF2.find(valor[0])
                    sNum = str(pos) + valor[1:-1]
                    num=int(sNum)
                elif valor[0].isdigit():
                    num=int(valor[:-1])
                if num!=None and TABLA_NIF[num%23] == valor[-1]:
                    bRet=True
        except:
            pass
    return bRet

def validarCIF(valor):
    CLAVES_CIF='PQS' + 'ABEH' + 'CDFGJRUVNW'
    CLAVES_NIF1 = 'LKM'                       # Son especiales, se validan
                                              # como CIFs
    CLAVES_NIF2 = 'XYZ'
    CLAVES_NIF = CLAVES_NIF1 + CLAVES_NIF2


    CONTROL_CIF_LETRA = 'KPQS'
    CONTROL_CIF_NUMERO = 'ABEH'

    EQUIVALENCIAS_CIF = {1:'A',
                         2:'B',
                         3:'C',
                         4:'D',
                         5:'E',
                         6:'F',
                         7:'G',
                         8:'H',
                         9:'I',
                         10:'J',
                         0:'J'}
    """
    Nos indica si un CIF es valido.
    El valor debe estar normalizado
    @note:
    - ante cualquier problema se valida como False
    """
    bRet = False
    if len(valor) == 9:
        v0 = valor[0]
        if v0 in CLAVES_NIF1 or v0 in CLAVES_CIF:
            try:
                sumPar = 0
                sumImpar = 0
                for i in xrange(1,8):
                    if i % 2:
                        v = int(valor[i]) * 2
                        if v > 9: v = 1 + (v - 10)
                        sumImpar += v
                    else:
                        v = int(valor[i])
                        sumPar += v
                suma = sumPar + sumImpar
                e = suma % 10
                d = 10 - e
                letraCif = EQUIVALENCIAS_CIF[d]
                if valor[0] in CONTROL_CIF_LETRA:
                    if valor[-1] == letraCif:
                        bRet = True
                elif valor[0] in CONTROL_CIF_NUMERO:
                    
                    if d == 10: 
                        d = 0
                    if valor[-1] == str(d):
                        bRet = True
                else:
                    if d == 10: d = 0
                    if valor[-1] == str(d) or valor[-1] == letraCif:
                        bRet = True
            except:
                pass
    return bRet

def validar(valor):
    CLAVES_NIF1 = 'LKM'                       # Son especiales, se validan
                                              # como CIFs
    CLAVES_NIF2 = 'XYZ'
    CLAVES_NIF = CLAVES_NIF1 + CLAVES_NIF2


    """
    Nos valida un CIF o un NIF
    """
    bRet = False
    if len(valor) == 9:
        if valor[0] in CLAVES_NIF or valor[0].isdigit():
            bRet = validarNIF(valor)
        else:
            bRet = validarCIF(valor)

    return bRet

UDF_validar = udf(validar, StringType())


def compararNIF(Tabla_A, Campo_A, Tabla_B, Campo_B):
    print(Tabla_A.count())
    print(Tabla_B.count())
    
    Nifs = Tabla_A.select(Campo_A).withColumnRenamed(Campo_A, 'Identificador').union(Tabla_B.select(Campo_B).withColumnRenamed(Campo_B,'Identificador')).withColumn('Nif_valido', UDF_validar(F.col('Identificador')))
    Nifs_ok = Nifs.where(F.col('Nif_valido') == True)
    Nifs_ko = Nifs.where(F.col('Nif_valido') == False)
    
    inner = Tabla_A.select(Campo_A).join(Tabla_B.select(Campo_B).withColumnRenamed(Campo_B, Campo_A),
                                         [Campo_A],
                                         'inner').drop_duplicates().withColumn('Inner', F.lit(1))
    
    inner = inner.withColumn(Campo_A, F.when(F.col(Campo_A) != 'NaN', F.col(Campo_A)).otherwise(F.lit(None)))

    Tabla_A = Tabla_A.join(inner, [Campo_A], 'left')
    Result_coinciden = Tabla_A.where(F.col('inner').isNotNull()).drop('inner')
    Result_no_coinciden = Tabla_A.where(F.col('inner').isNull()).drop('inner')
    
    return Result_coinciden, Result_no_coinciden, Nifs_ok, Nifs_ko



def Distance_ratcliff_obershelp(column1, column2):
    return textdistance.ratcliff_obershelp(column1, column2)

udf_Distance_ratcliff_obershelp = F.udf(Distance_ratcliff_obershelp, FloatType())

def calcular_Distance_ratcliff_obershelp(spark, Tabla_A, Tabla_B, Campo_A, Campo_B):
    schema = StructType([
        StructField('Campo_A', StringType(), True),
        StructField('Campo_B', StringType(), True),
        StructField('Distance', FloatType(), True)])
        
    print(Tabla_A.count())
    print(Tabla_B.count())
    resultados = []
    
    for Nombre in tqdm(Tabla_A.select(F.col(Campo_A)).drop_duplicates().collect()):
        distancias = Tabla_B.select(F.col(Campo_B)).withColumn('Distance', udf_Distance_ratcliff_obershelp(F.col(Campo_B), F.lit(Nombre[0])))
        Resultados_vuelta = distancias.where(F.col('Distance') == distancias.groupby().max('Distance').collect()[0][0]).withColumn('Nombre_2', F.lit(Nombre[0])).collect()
        resultados.append([Resultados_vuelta[0][0], Resultados_vuelta[0][2], Resultados_vuelta[0][1]])
    
    return spark.createDataFrame(resultados, schema)

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def data_conteos(Conteos, Nombre, Fichero, bloque, Fecha):
    return Conteos.append([Nombre, Fichero.count(),  Fichero.columns, bloque, Fecha])


# Leemos los ficheros:
#------------------------


def index_es(spark,df,target_names_column,index_name):
    '''
    inputs:
        df : pd.DataFrame object containing the two columns source names and target names
        target_names : type: str, represents the the target names column
        index_name : type: str, represents the index of elastic search object
    output:
        return elasticsearch object indexed, and initialized with target names
    
    '''
    #print('df')
    #print(df)
    
    #print('target_names_column')
    #print(target_names_column)
    
    #print('df[target_names_column]')
    #print(df[target_names_column])
    #df[target_names_column] = df[target_names_column].apply(lambda x: unidecode(x))
    target_names = [name.lower() for name in list(df[target_names_column])]
    es = Elasticsearch()
    
    for i in list(es.indices.get_alias("*").items()):
        es.indices.delete(index=list(i)[0])
    
    es.indices.delete(index='test-index', ignore=[400, 404])
    
    
    i=0
    for name in tqdm(target_names):
        doc = {'entity':name}
        es.index(index=index_name, id=i, document=doc)
        i+=1    
    return es


def remove_irrelevant_tokens(entity,tokens=['s l','s a u','s a','sa','sl']):
    for i in tokens:
        entity = re.sub(rf"\b{i}\b",'',entity.lower()).strip()
        #s = '\b'+i+'\b'
    
    return entity
                                     

def feature_extractor(entity1,entity2,fuzzy_threshold=0.8,nickname_threshold=0.46,
                      fuzzy_nickname_threshold=0.3):
    entity1 =  remove_irrelevant_tokens(entity1)
    entity2 =  remove_irrelevant_tokens(entity2)
    tokens1 = word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", " ",entity1.lower()))
    tokens2 = word_tokenize(re.sub(r"[^a-zA-Z0-9 ]", " ",entity2.lower()))
    len1 = len(tokens1)
    len2 = len(tokens2)
    M = np.min([len1,len2])
    N = np.max([len1,len2])
    mat = np.zeros((M,N))
    if len1 <= len2:
        list1 = tokens1
        list2 = tokens2
    else:
        list1= tokens2
        list2 = tokens1
    retained_rows=[]
    retained_cols=[]
    for m,i in enumerate(list1):
        for n,j in enumerate(list2):
            s = textdistance.ratcliff_obershelp(i,j)
            mat[m,n] = s
            if s==1:
                retained_rows.append(m)
                retained_cols.append(n)
    
    for m,i in enumerate(list1):
        if m not in retained_rows:
            for n,j in enumerate(list2):
                if n not in retained_cols:
                    if len(j) == 1 and i[0] == j and len(i) !=1:
                        #print((m,n))
                        mat[m,n] = 10000
                        retained_cols.append(n)

    max_rows = np.amax(mat,axis=1)
    n1 = len(max_rows[max_rows==1])
    n2 = len(max_rows[(max_rows<1) & (max_rows>=fuzzy_threshold)])
    n3 = 0#len(mat[(mat<fuzzy_threshold) & (mat>=nickname_threshold)])
    n4 = 0#len(mat[(mat<nickname_threshold) & (mat>=fuzzy_nickname_threshold)])
    n5  = len(max_rows[max_rows==10000])
    feature_vector = np.array([n1,n2,n5,M,N])
    return feature_vector


def compute_score(feature_vector,weights_vector=np.array([1,0.9,0.6])):
    match_features = feature_vector[:3]
    M = feature_vector[3]
    score = np.dot(match_features,weights_vector) / M
    return score

def detect_cities(entity, cities):
    entity = entity.lower()
    detected_city = None
    for city in cities:
        if city in entity:
            detected_city = city
            break
    return detected_city

def is_similar(city1,city2,syn_cities):
    if {city1,city2} in syn_cities:
        return True
    else:
        return False

def compare_names(name1, name2, cities, syn_cities):
    detected_city1 = detect_cities(name1, cities)
    detected_city2 = detect_cities(name2, cities)
    if detected_city1 and detected_city2 and detected_city1 != detected_city2 and not is_similar(detected_city1,detected_city2, syn_cities):
        score = 0
        #d_c = True
    else:
        feature_vector = feature_extractor(name1,name2)
        score = compute_score(feature_vector)
        #d_c = False
    return score#,d_c





def compare_loc(loc1,loc2,no_loc_list=[77, 88, 99]):
    try:
        loc1 = float(loc1)
    except:
        loc1=0
    try:
        loc2 = float(loc2)
    except:
        loc2=0
        
    max_range = 52
    min_range = 1
    if min_range <= loc1 <= max_range and  min_range <= loc2 <= max_range:
        if loc1 == loc2:
            return 1
        else:
            return 0
    else:
        if loc1 in no_loc_list or loc2 in no_loc_list:
            return 0.5
        else:
            return 0

        
        
def format_results(results):
    names_source = []
    names_target = []
    cities_source=[]
    cities_target=[]
    mun_source=[]
    mun_target=[]
    countries_source=[]
    countries_target=[]
    for i in results:
        source_name = results[i]['source_name']
        source_city = results[i]['source_city']
        source_mun = results[i]['source_municipality']
        source_country = results[i]['source_country']
        for element in results[i]['matching_names_by_elasticsearch']:
            target_name = element[1]
            target_city = element[2]
            target_mun = element[3]
            target_country = element[4]
            names_source.append(source_name)
            cities_source.append(source_city)
            mun_source.append(source_mun)
            countries_source.append(source_country)
            names_target.append(target_name)
            cities_target.append(target_city)
            mun_target.append(target_mun)
            countries_target.append(target_country)
            
    result = pd.DataFrame({'source_name':names_source,
             'target_name':names_target,
             'source_city':cities_source,
             'target_city':cities_target,
             'source_mun':mun_source,
             'target_mun':mun_target,
             'source_country':countries_source,
             'target_country':countries_target})
            
    return result

def compute_scores(cities, syn_cities, results,name_weight,city_weight,mun_weight,country_weight):
    results['name_score'] = results.apply(lambda x: compare_names(x.source_name, x.target_name, cities, syn_cities), axis=1)
    results['city_score'] = results.apply(lambda x: compare_loc(x.source_city, x.target_city), axis=1)
    results['mun_score'] = results.apply(lambda x: compare_loc(x.source_mun, x.target_mun), axis=1)
    results['country_score'] = results.apply(lambda x: compare_loc(x.source_country, x.target_country), axis=1)
    #print(results['name_score'])
    results['weighted_name_score'] = [i*name_weight for i in results['name_score']]
    results['weighted_city_score'] = [i*city_weight for i in results['city_score']]
    
    results['weighted_mun_score'] = [i*mun_weight for i in results['mun_score']]
    results['weighted_country_score'] = [i*country_weight for i in results['country_score']]
    results['final_score'] = [(a+b+c+d)/(name_weight+city_weight+mun_weight+country_weight) for a,b,c,d in zip(results['weighted_name_score'],
                                                                                                             results['weighted_city_score'],
                                                                                                              results['weighted_mun_score'],
                                                                                                              results['weighted_country_score']
                                                                                                             )]
    return results

# def get_idx_to_entity(df,entities_column):
#     entities = list(df[entities_column])
#     idx_to_entity = {i:j for i,j in enumerate(entities)}
#     return idx_to_entity

def get_matching_entities_by_elasticSearch(spark,
                                           syn_cities,
                                           cities,
                                           df1,
                                           source_names_column,
                                           city_column_source,
                                           municipality_column_source,
                                           country_column_source,
                                           df2,
                                           target_names_column,
                                           city_column_target,                                          
                                           municipality_column_target,                                           
                                           country_column_target,
                                           index_name,
                                           name_weight=0.93283582,city_weight=0.06716418,
                                           mun_weight=0.0,country_weight=0.0,
                                           threshold = 0.9141791,
                                           size=50):
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
    es = index_es(spark,df2,target_names_column,index_name)
    source_names = list(df1[source_names_column])
    cities_source=list(df1[city_column_source])
    cities_target=list(df2[city_column_target])
    mun_source=list(df1[municipality_column_source])
    mun_target=list(df2[municipality_column_target])
    countries_source=list(df1[country_column_source])
    countries_target=list(df2[country_column_target])
    
    source_idx=0
    results = {}
    for entity_i in tqdm(source_names):
        matching_entities = []
        query = {"from": 0,
                 "size": 10,
                 "query": {"match": {"entity":{'query':entity_i,'fuzziness':'AUTO'}}},
                 "sort":[{'_score':{'order':'desc'}}]}
        resp = es.search(index=index_name, body=query)
        for element in resp['hits']['hits']:
            target_idx = int(element['_id'])
            entity_j = element['_source']['entity']
            matching_entities.append((target_idx,entity_j,cities_target[target_idx],mun_target[target_idx],countries_target[target_idx]))
            
        results[source_idx] = {'source_name':entity_i,
                               'source_city':cities_source[source_idx],
                               'source_municipality':mun_source[source_idx],
                               'source_country':countries_source[source_idx],
                            'matching_names_by_elasticsearch':matching_entities}
        source_idx +=1
    results = format_results(results)
    #print(results)
    results = compute_scores(cities, syn_cities, results,name_weight,city_weight,mun_weight,country_weight)
    #print('threshold', threshold)
    #results = results[results['final_score']>=threshold]
    
    return results





def calcular_Distance_ratcliff_obershelp_ElasticSearch_Ubicacion(spark,
                                                                 syn_cities,
                                                                 cities,
                                                                 df1,
                                                                 source_names,
                                                                 city_column_source,
                                                                 source_municipality,
                                                                 country_column_source,
                                                                 df2,
                                                                 target_names,
                                                                 city_column_target,                                          
                                                                 target_municipality,                                           
                                                                 country_column_target,
                                                                 index_name,
                                                                 name_weight=0.93283582,
                                                                 city_weight=0.06716418,
                                                                 mun_weight=0.0,
                                                                 country_weight=0.0,
                                                                 threshold = 0.9141791,
                                                                 size=50):
    
    #print('df1')
    #print(df1)
    
    #print('source_names')
    #print(source_names)
    
    #print('source_municipality')
    #print(source_municipality)
    print("entra funcion comparacion")
    df_1 = df1.select(source_names, city_column_source).toPandas()
    df_2 = df2.select(target_names, city_column_target).toPandas()
    for column in ['source_municipality', 'source_country']:
        df_1[column] = 0
    for column in ['target_municipality', 'target_country']:
        df_2[column] = 0
        
    #print('df_1')
    #print(df_1)
    #print('df_1')
    #print(df_2)
    results = get_matching_entities_by_elasticSearch(spark,
                                                     syn_cities,
                                                     cities,
                                                     df_1,
                                                     source_names,
                                                     city_column_source,
                                                     'source_municipality',
                                                     'source_country',
                                                     df_2,
                                                     target_names,
                                                     city_column_target,                                          
                                                     'target_municipality',                                           
                                                     'target_country',
                                                     index_name='index_name',
                                                     name_weight=0.93283582,city_weight=0.06716418,
                                                     mun_weight=0.0,country_weight=0.0,
                                                     threshold = 0.9141791,
                                                     size=size)
    #print('Final')
    #print(results)
    
    
    Lista_Diferentes = [['Alicante'],
                    ['Asturias'],
                    ['Cádiz'],
                    ['Madrid'],
                    ['Barcelona', 'Catalunya'],
                    ['Valencia'],
                    ['Sevilla'],
                    ['Málaga'],
                    ['Murcia'],
                    ['Baleares'],
                    ['Las Palmas'],
                    ['Vizcaya'],
                    ['A Coruña', 'Coruna'],
                    ['Santa Cruz de Tenerife'],
                    ['Zaragoza'],
                    ['Pontevedra'],
                    ['Granada'],
                    ['Tarragona'],
                    ['Córdoba'],
                    ['Gerona'],
                    ['Almería'],
                    ['Guipúzcoa'],
                    ['Toledo'],
                    ['Badajoz'],
                    ['Navarra'],
                    ['Jaén'],
                    ['Cantabria'],
                    ['Castellón'],
                    ['Huelva'],
                    ['Valladolid'],
                    ['Ciudad Real'],
                    ['León'],
                    ['Lérida'],
                    ['Albacete'],
                    ['Cáceres'],
                    ['Burgos'],
                    ['Álava'],
                    ['Salamanca'],
                    ['Lugo'],
                    ['La Rioja'],
                    ['Orense'],
                    ['Guadalajara'],
                    ['Huesca'],
                    ['Cuenca'],
                    ['Zamora'],
                    ['Ávila'],
                    ['Palencia'],
                    ['Segovia'],
                    ['Teruel'],
                    ['Soria'],
                    ['Melilla'],
                    ['Ceuta'],
                    ['España', 'Spain'],
                    ['Estonia'],
                    ['Italia'],
                    ['Galicia'],
                    ['Bulgaria'],
                    ['Brasil', 'Brazil']]
    
    #print(results)
    print("entra funcion comparacion 2")
    for i_num_linea in range(len(Lista_Diferentes)):
        for j_num_linea in range(i_num_linea + 1, len(Lista_Diferentes)):
            for i in Lista_Diferentes[i_num_linea]:
                for j in Lista_Diferentes[j_num_linea]:

                    i = pre_process_normalize(i)
                    j = pre_process_normalize(j)
                    #print(i_num_linea, j_num_linea, i, ',', j, results.shape)

                    results = results[~((((results['source_name'].str.contains(i)) & (results['target_name'].str.contains(j))) |
                                                   ((results['source_name'].str.contains(j)) & (results['target_name'].str.contains(i)))) &
                                                  (~((results['source_name'].str.contains(i)) & (results['source_name'].str.contains(j)))) &
                                                  (~((results['target_name'].str.contains(i)) & (results['target_name'].str.contains(j)))))
                                          ]
    print("entra funcion comparacion 2")
    schema = StructType([
        StructField('source_names', StringType(), True),
        StructField('target_names', StringType(), True),
        StructField('source_municipality', StringType(), True), #Le cambio el orden
        StructField('target_municipality', StringType(), True), #Le cambio el orden
        StructField('source_city', StringType(), True), #Le cambio el orden
        StructField('target_city', StringType(), True), #Le cambio el orden
        StructField('source_country', StringType(), True),                     
        StructField('target_country', StringType(), True),
        StructField('name_score', StringType(), True),
        StructField('city_score', StringType(), True),
        StructField('mun_score', StringType(), True),
        StructField('country_score', StringType(), True),
        StructField('weighted_name_score', StringType(), True),
        StructField('weighted_city_score', StringType(), True),
        StructField('weighted_mun_score', StringType(), True),
        StructField('weighted_country_score', StringType(), True),
        StructField('final_score', FloatType(), True)])

    results = spark.createDataFrame(results, schema)
    
    stopwords_es = nltk.corpus.stopwords.words('spanish')
    
    
    results = results.withColumn("aux", F.split("source_names", "\\s+"))
    remover = StopWordsRemover(stopWords=stopwords_es, inputCol="aux", outputCol="source_names_stopwords")
    results = remover.transform(results).withColumn("source_names_stopwords", F.array_join("source_names_stopwords", " "))

    results = results.withColumn("aux", F.split("target_names", "\\s+"))
    remover = StopWordsRemover(stopWords=stopwords_es, inputCol="aux", outputCol="target_names_stopwords")
    results = remover.transform(results).withColumn("target_names_stopwords", F.array_join("target_names_stopwords", " "))

        
    #return spark.createDataFrame(results, schema).withColumn('final_score', 0.9*(udf_Distance_ratcliff_obershelp(F.col('source_names'), F.col('target_names'))) + F.col('weighted_city_score'))
    #print("entra al ratcliff")
    return results.withColumn('final_score', udf_Distance_ratcliff_obershelp(F.col('source_names_stopwords'), F.col('target_names_stopwords'))).drop('aux').withColumn('final_score', F.when(F.col('source_names_stopwords') != (F.col('target_names_stopwords')), F.col('final_score')).otherwise(F.lit(1))).drop_duplicates()
    #print("sale del ratcliff")





#def calcular_Distance_ratcliff_obershelp_ElasticSearch(spark, Tabla_A, Tabla_B, Campo_A, Campo_B, index_name):
#    es = index_es(Tabla_A, Campo_A, index_name)
#    return get_matching_entities_by_elasticSearch(spark, es, Tabla_B, Campo_B, index_name, size=20)



def bigrdd_topandas(df, num) :
    iters = int(df.count()/num)+1
    df_ = pd.DataFrame()
    df = df.withColumn("Row",F.row_number().over(Window().orderBy(F.lit(0))))
    for it in range(0, iters):
#        print(it)
        df_tmp = df.filter(F.col("Row").between(it * num, (it+1) * num -1)).drop("Row").toPandas()
        df_ = pd.concat([df_, df_tmp]).reset_index(drop = True)
       
    return df_

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



# Funcion que aplica las tecnicas de limpieza
def normalizarColumnas(texto):
    """
    Ejemplo de uso:
    -----------------
    df_final = df.withColumn("NombreColOutput",UDF_normalizarTexto("NombreCol2Normalizar"))
    """
    # 1.1. Eliminar tildes
    texto=unidecode.unidecode(texto)
    # 1.2. Elimina caracteres especiales y digitos lo reemplaza por un espacio en Blanco
    texto=re.sub("(\\W)+"," ",texto)
    # 1.3. Eliminar la concatenación de espacios en blanco (quedarnos únicamente con un espacio en blanco entre palabras)
    texto=" ".join(texto.split())
    # 1.4. Pasar a minúsculas:
    #texto = texto.lower()
    
    return texto

def fudf(val):
    return reduce (lambda x, y:x+y, val)

flattenUdf = F.udf(fudf, T.ArrayType(T.IntegerType()))




def exportar_cruzan_CIF(Ruta_Output,
                       dataframe1_output_cruzan_cruce,
                       df_name_dataframe1_output_cruzan_cruce,
                       Dataframe_uo_orig,
                       ids_maestro):

    print(df_name_dataframe1_output_cruzan_cruce)

    dataframe1_output_cruzan_cruce = Dataframe_uo_orig.join(dataframe1_output_cruzan_cruce,
                                                             (Dataframe_uo_orig.Entidad_Norm == dataframe1_output_cruzan_cruce.Entidad_Norm) &
                                                             (Dataframe_uo_orig.Provincia_Entidad == dataframe1_output_cruzan_cruce.Provincia_Entidad) &
                                                             (Dataframe_uo_orig.CIF == dataframe1_output_cruzan_cruce.CIF),
                                                             'inner').drop(dataframe1_output_cruzan_cruce.Entidad_Norm).drop(dataframe1_output_cruzan_cruce.Provincia_Entidad).drop(dataframe1_output_cruzan_cruce.CIF)

    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'nan', F.col('Provincia_Entidad')).otherwise(F.lit(' ')))

    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(' ')))
    
    dataframe1_output_cruzan_cruce= dataframe1_output_cruzan_cruce.withColumn("Provincia_Match",col("Provincia_Match").cast("integer"))

    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Match', F.when(F.col('Provincia_Match').isNotNull(), F.col('Provincia_Match')).otherwise(F.lit(' ')))
    
    dataframe1_output_cruzan_cruce = \
        dataframe1_output_cruzan_cruce.join(ids_maestro,
                                             (ids_maestro.CIF == dataframe1_output_cruzan_cruce.CIF) &
                                             (ids_maestro.Entidad_Norm == dataframe1_output_cruzan_cruce.Entidad_Match) &
                                             (ids_maestro.Provincia_Entidad == dataframe1_output_cruzan_cruce.Provincia_Match),
                                             'inner') \
    .drop(ids_maestro.Entidad_Norm)\
    .drop(ids_maestro.Provincia_Entidad)\
    .drop(ids_maestro.CIF).persist()
    
    #dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Match', F.when(F.col('Provincia_Match') != ' ', F.col('Provincia_Match')).otherwise(None))
    #dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(None))

    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.select('Id',
                                                                             'Entidad',
                                                                             'Origen_Solicitud',
                                                                             'Identificadores_Origen',
                                                                             'Entidad_Norm',
                                                                             'CIF',
                                                                             'CIF_validacion',
                                                                             'PIC',
                                                                             'Tipo',
                                                                             'Provincia_Entidad',
                                                                             'Pais_Entidad',
                                                                             'Centro',
                                                                             'Centro_Norm',
                                                                             'Provincia_Centro',
                                                                             'Entidad_Match',
                                                                             'Provincia_Match',
                                                                             'Match')

    #Casteamos columnas a entero
    #dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad',dataframe1_output_cruzan_cruce.Provincia_Entidad.cast(IntegerType()))
    #dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Match',dataframe1_output_cruzan_cruce.Provincia_Match.cast(IntegerType()))

    #Guardamos el fichero
    dataframe1_output_cruzan_cruce.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe1_output_cruzan_cruce)
    dataframe1_output_cruzan_cruce = spark.read.parquet(Ruta_Output + df_name_dataframe1_output_cruzan_cruce)

    #Creamos el fichero en Pandas
    pd_dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.dropDuplicates().toPandas()

    #Casteamos las columnas
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe1_output_cruzan_cruce.columns:
            pd_dataframe1_output_cruzan_cruce[column] = np.where(pd_dataframe1_output_cruzan_cruce[column]==' ', np.nan, pd_dataframe1_output_cruzan_cruce[column])
            pd_dataframe1_output_cruzan_cruce[column] = pd_dataframe1_output_cruzan_cruce[column].astype('float').astype('Int64')

    #Guardamos el fichero en csv
    pd_dataframe1_output_cruzan_cruce.to_csv(Ruta_Output + df_name_dataframe1_output_cruzan_cruce + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    print(dataframe1_output_cruzan_cruce.count())
    dataframe1_output_cruzan_cruce.show()
    
    print(pd_dataframe1_output_cruzan_cruce)

    
    
def exportar_no_cruzan(Ruta_Output,
                    dataframe2_output_nocruzan_cruce,
                    df_name_dataframe2_output_nocruzan_cruce,
                    Dataframe_uo_orig):

    print(df_name_dataframe2_output_nocruzan_cruce)

    #Añadimos la informacion de las solicitudes al fichero
    dataframe2_output_nocruzan_cruce = Dataframe_uo_orig.join(dataframe2_output_nocruzan_cruce,
                                                             (Dataframe_uo_orig.Entidad_Norm == dataframe2_output_nocruzan_cruce.Entidad_Norm) &
                                                             (Dataframe_uo_orig.Provincia_Entidad == dataframe2_output_nocruzan_cruce.Provincia_Entidad) &
                                                             (Dataframe_uo_orig.CIF == dataframe2_output_nocruzan_cruce.CIF),
                                                             'inner').drop(dataframe2_output_nocruzan_cruce.Entidad_Norm).drop(dataframe2_output_nocruzan_cruce.Provincia_Entidad).drop(dataframe2_output_nocruzan_cruce.CIF)

    #Seleccionamos el orden de las columnas del fichero
    dataframe2_output_nocruzan_cruce = dataframe2_output_nocruzan_cruce.select('Entidad',
                                                                                 'Origen_Solicitud',
                                                                                 'Identificadores_Origen',
                                                                                 'Entidad_Norm',
                                                                                 'CIF',
                                                                                 'CIF_validacion',
                                                                                 'PIC',
                                                                                 'Tipo',
                                                                                 'Provincia_Entidad',
                                                                                 'Pais_Entidad',
                                                                                 'Centro',
                                                                                 'Centro_Norm',
                                                                                 'Provincia_Centro')
    
    #dataframe2_output_nocruzan_cruce = dataframe2_output_nocruzan_cruce.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    #dataframe2_output_nocruzan_cruce = dataframe2_output_nocruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(' '))
    #dataframe2_output_nocruzan_cruce = dataframe2_output_nocruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(' '))
    
    #Casteamos columnas a entero
    dataframe2_output_nocruzan_cruce = dataframe2_output_nocruzan_cruce.withColumn('Provincia_Entidad',dataframe2_output_nocruzan_cruce.Provincia_Entidad.cast(IntegerType()))

    #Guardamos el fichero
    dataframe2_output_nocruzan_cruce.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe2_output_nocruzan_cruce)
    dataframe2_output_nocruzan_cruce = spark.read.parquet(Ruta_Output + df_name_dataframe2_output_nocruzan_cruce)

    #Creamos el fichero en Pandas
    pd_dataframe2_output_nocruzan_cruce = dataframe2_output_nocruzan_cruce.dropDuplicates().toPandas()

    #Casteamos las columnas
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe2_output_nocruzan_cruce.columns:
            pd_dataframe2_output_nocruzan_cruce[column] = np.where(pd_dataframe2_output_nocruzan_cruce[column]==' ', np.nan, pd_dataframe2_output_nocruzan_cruce[column])
            pd_dataframe2_output_nocruzan_cruce[column] = pd_dataframe2_output_nocruzan_cruce[column].astype('float').astype('Int64')

    #Guardamos el fichero en csv
    pd_dataframe2_output_nocruzan_cruce.to_csv(Ruta_Output + df_name_dataframe2_output_nocruzan_cruce + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    
    
    
def exportar_cruzan_nombre(Ruta_Output,
                           dataframe1_output_cruzan_cruce,
                           df_name_dataframe1_output_cruzan_cruce,
                           Dataframe_uo_orig,
                           ids_maestro):   
    

    #Añadimos la informacion de las solicitudes al fichero
    dataframe1_output_cruzan_cruce = Dataframe_uo_orig.join(dataframe1_output_cruzan_cruce,
                                                             (Dataframe_uo_orig.Entidad_Norm == dataframe1_output_cruzan_cruce.Entidad_Norm) &
                                                             (Dataframe_uo_orig.Provincia_Entidad == dataframe1_output_cruzan_cruce.Provincia_Entidad) &
                                                             (Dataframe_uo_orig.CIF == dataframe1_output_cruzan_cruce.CIF),
                                                             'inner').drop(dataframe1_output_cruzan_cruce.Entidad_Norm).drop(dataframe1_output_cruzan_cruce.Provincia_Entidad).drop(dataframe1_output_cruzan_cruce.CIF)

    #Añadimos los ids
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn("Provincia_Match",col("Provincia_Match").cast("integer"))

    
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'nan', F.col('Provincia_Entidad')).otherwise(F.lit(' ')))

    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(' ')))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(' ')))
    
    
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Match', F.when(F.col('Provincia_Match').isNotNull(), F.col('Provincia_Match')).otherwise(F.lit(' ')))

    
    
    dataframe1_output_cruzan_cruce = \
        dataframe1_output_cruzan_cruce.join(ids_maestro.drop('CIF'),
                                             (ids_maestro.Entidad_Norm == dataframe1_output_cruzan_cruce.Entidad_Match) &
                                             (ids_maestro.Provincia_Entidad == dataframe1_output_cruzan_cruce.Provincia_Match),
                                             'inner') \
        .drop(ids_maestro.Entidad_Norm)\
        .drop(ids_maestro.Provincia_Entidad).persist()
    
    #Seleccionamos el orden de las columnas del fichero
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.select('Id',
                                                                             'Entidad',
                                                                             'Origen_Solicitud',
                                                                             'Identificadores_Origen',
                                                                             'Entidad_Norm',
                                                                             'CIF',
                                                                             'CIF_validacion',
                                                                             'PIC',
                                                                             'Tipo',
                                                                             'Provincia_Entidad',
                                                                             'Pais_Entidad',
                                                                             'Centro',
                                                                             'Centro_Norm',
                                                                             'Provincia_Centro',
                                                                             'Entidad_Match',
                                                                             'Provincia_Match',
                                                                             'final_score',
                                                                             'Match')
    

    #Casteamos columnas a entero
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Entidad',dataframe1_output_cruzan_cruce.Provincia_Entidad.cast(IntegerType()))
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Match',dataframe1_output_cruzan_cruce.Provincia_Match.cast(IntegerType()))
    
    dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.withColumn('Provincia_Match', F.when(F.col('Provincia_Match').isNotNull(), F.col('Provincia_Match')).otherwise(F.lit(' ')))

    #Guardamos el fichero
    dataframe1_output_cruzan_cruce.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe1_output_cruzan_cruce)
    dataframe1_output_cruzan_cruce = spark.read.parquet(Ruta_Output + df_name_dataframe1_output_cruzan_cruce)

    #Creamos el fichero en Pandas
    pd_dataframe1_output_cruzan_cruce = dataframe1_output_cruzan_cruce.dropDuplicates().toPandas()

    #Casteamos las columnas
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe1_output_cruzan_cruce.columns:
            pd_dataframe1_output_cruzan_cruce[column] = np.where(pd_dataframe1_output_cruzan_cruce[column]==' ', np.nan, pd_dataframe1_output_cruzan_cruce[column])
            pd_dataframe1_output_cruzan_cruce[column] = pd_dataframe1_output_cruzan_cruce[column].astype('float').astype('Int64')

    #Guardamos el fichero en csv
    pd_dataframe1_output_cruzan_cruce.to_csv(Ruta_Output + df_name_dataframe1_output_cruzan_cruce + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    print(dataframe1_output_cruzan_cruce.count())
    dataframe1_output_cruzan_cruce.show()
    
    print(pd_dataframe1_output_cruzan_cruce)
    
    
def exportar_csv(spark, Ruta_Output, fichero, nombre):
    #Creamos el fichero pandas
    pd_fichero = fichero.dropDuplicates().toPandas()

    #Casteamos a enteros los valores necesarios
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_fichero.columns:
            pd_fichero[column] = np.where(pd_fichero[column]==' ', np.nan, pd_fichero[column])
            pd_fichero[column] = pd_fichero[column].astype('float').astype('Int64')

    #Eliminamos los saltos de carro
    pd_fichero = pd_fichero.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_fichero.to_csv(Ruta_Output + nombre + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    
    
    
    
    
    
    
    
    
def Funcion_Entidades(Bloque,
                      spark,
                      Threshold,
                      Ruta_Input,
                      Ruta_Output,
                      dataframe1_output_cruzan_cruce1,
                      dataframe2_output_nocruzan_cruce1,
                      dataframe1_output_cruzan_cruce2,
                      dataframe2_output_nocruzan_cruce2,
                      dataframe1_output_cruzan_cruce3,
                      dataframe2_output_nocruzan_cruce3,
                      dataframe1_output_cruzan_cruce4,
                      dataframe2_output_nocruzan_cruce4,
                      dataframe1_output_cruzan_cruce5,
                      dataframe2_output_nocruzan_cruce5,
                      dataframe1_output_cruzan_cruce6,
                      dataframe2_output_nocruzan_cruce6,
                      dataframe_output_unificacion1,
                      dataframe_output_unificacion2,
                      dataframe_output_unificacion3,
                      dataframe_output_unificacion4,
                      dataframe_output_unificacion5,
                      dataframe_output_unificacion6,
                      Dataframe_nueva_invente_nif,
                      Dataframe_nueva_invente_nombre,
                      Dataframe_nueva_DIR3_nif,
                      Dataframe_nueva_DIR3_nombre,
                      Dataframe_maestro,
                      df_name_Dataframe_maestro,
                      Dataframe_uo,
                      df_name_Dataframe_uo,
                      Dataframe_invente,
                      df_name_Dataframe_invente,
                      Dataframe_DIR3,
                      df_name_Dataframe_DIR3,
                      Info_Ids_ant):

    
    #Iniciamos la funcion
    #Cambio explicacion funcional de la funcion
    print("Iniciamos funcion 'Entidades'")
    
    print('Bloque: ', Bloque)
    print('Threshold: ', Threshold)

    #Mostramos conteos
    print('Solicitudes- Registros: ', Dataframe_uo.count())
    print('Maestro - Registros: ', Dataframe_maestro.count())
    print('INVENTE - Registros: ', Dataframe_invente.count())
    print('DIR3 - Registros: ', Dataframe_DIR3.count())


    for column in ['Entidad', 'Origen_Solicitud', 'Identificadores_Origen', 'Entidad_Norm', 'CIF', 'CIF_validacion', 'PIC', 'Tipo', 'Provincia_Entidad', 'Pais_Entidad', 'Centro', 'Centro_Norm', 'Provincia_Centro', 'Tipo_Persona']:
        if(column not in Dataframe_uo.columns):
            Dataframe_uo = Dataframe_uo.withColumn(column, F.lit(None).cast(StringType()))
            print('Añadimos la columna ', column, ' en blanco al fichero')
    

    

    
    #Seleccionamos las variables originales del maestro inicial
    
    Dataframe_uo = Dataframe_uo.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    
    Dataframe_uo = Dataframe_uo.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    Dataframe_maestro = Dataframe_maestro.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    Dataframe_invente = Dataframe_invente.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    Dataframe_DIR3 = Dataframe_DIR3.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    
    Dataframe_maestro = Dataframe_maestro.select('Id', 'ID_ENTIDAD', 'NIF_COD', 'ACRONIMO', 'NOMBRE_ENTIDAD', 'Nombre_Entidad_Mostrar', 'TIPO_ENTIDAD_N1_1', 'TIPO_ENTIDAD_N2_1', 'DIRECCION_POSTAL', 'COD_POSTAL', 'COD_PROVINCIA', 'PROVINCIA', 'COD_CCAA', 'CCAA', 'ENLACE_WEB', 'SOMMA', 'TIPO_ENTIDAD_REGIONAL', 'ESTADO_x', 'Entidad_Norm', 'CIF', 'Provincia_Entidad')
    
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


    #Realizamos una copia original de los ficheros, ya que más adelante los vamos a modificar y nos interesa tener el original al final
    Dataframe_uo_orig = Dataframe_uo
    Dataframe_maestro_orig = Dataframe_maestro
    Dataframe_invente_orig = Dataframe_invente
    Dataframe_DIR3_orig = Dataframe_DIR3
    
    

    Dataframe_DIR3_orig = Dataframe_DIR3_orig.drop('Unnamed: 0')
    
    #ahora para el cruce del nif quieren el nivel jerarquico
    DIR3_aux = Dataframe_DIR3.select('Entidad_Norm', 'CIF', 'Provincia_Entidad', 'N_NIVEL_JERARQUICO')


    
    #Si no tenemos el campo Id, lo creamos en blanco.
    #Cambio, revisar si esto se puede eliminar
    if('Id' not in Dataframe_invente.columns):
        Dataframe_invente = Dataframe_invente.withColumn('Id', F.lit(None).cast(StringType()))

    if('Id' not in Dataframe_DIR3.columns):
        Dataframe_DIR3 = Dataframe_DIR3.withColumn('Id', F.lit(None).cast(StringType()))

    #Seleccionamos las columnas que utilizaremos en el proceso
    Dataframe_uo = Dataframe_uo.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').dropDuplicates()
    Dataframe_maestro = Dataframe_maestro.select('Id', 'CIF', 'Entidad_Norm', 'Provincia_Entidad').dropDuplicates()
    Dataframe_invente = Dataframe_invente.select('Id', 'CIF', 'Entidad_Norm', 'Provincia_Entidad').dropDuplicates()
    Dataframe_DIR3 = Dataframe_DIR3.select('Id', 'CIF', 'Entidad_Norm', 'Provincia_Entidad').dropDuplicates()


    #Casteamos las columnas necesarias a StringType
    Dataframe_uo = Dataframe_uo.withColumn('Provincia_Entidad',Dataframe_uo.Provincia_Entidad.cast(StringType()))
    Dataframe_maestro = Dataframe_maestro.withColumn('Provincia_Entidad',Dataframe_maestro.Provincia_Entidad.cast(StringType()))
    Dataframe_invente = Dataframe_invente.withColumn('Provincia_Entidad',Dataframe_invente.Provincia_Entidad.cast(StringType()))
    Dataframe_DIR3 = Dataframe_DIR3.withColumn('Provincia_Entidad',Dataframe_DIR3.Provincia_Entidad.cast(StringType()))

    #Casteamos las columnas necesarias a StringType
    Dataframe_uo_orig = Dataframe_uo_orig.withColumn('Provincia_Entidad',Dataframe_uo_orig.Provincia_Entidad.cast(StringType()))
    Dataframe_maestro_orig = Dataframe_maestro_orig.withColumn('Provincia_Entidad',Dataframe_maestro_orig.Provincia_Entidad.cast(StringType()))
    Dataframe_invente_orig = Dataframe_invente_orig.withColumn('Provincia_Entidad',Dataframe_invente_orig.Provincia_Entidad.cast(StringType()))
    Dataframe_DIR3_orig = Dataframe_DIR3_orig.withColumn('Provincia_Entidad',Dataframe_DIR3_orig.Provincia_Entidad.cast(StringType()))




    df_name_dataframe1_output_cruzan_cruce1 = dataframe1_output_cruzan_cruce1
    df_name_dataframe2_output_nocruzan_cruce1 = dataframe2_output_nocruzan_cruce1
    df_name_dataframe1_output_cruzan_cruce2 = dataframe1_output_cruzan_cruce2
    df_name_dataframe2_output_nocruzan_cruce2 = dataframe2_output_nocruzan_cruce2
    df_name_dataframe1_output_cruzan_cruce3 = dataframe1_output_cruzan_cruce3
    df_name_dataframe2_output_nocruzan_cruce3 = dataframe2_output_nocruzan_cruce3
    df_name_dataframe1_output_cruzan_cruce4 = dataframe1_output_cruzan_cruce4
    df_name_dataframe2_output_nocruzan_cruce4 = dataframe2_output_nocruzan_cruce4
    df_name_dataframe1_output_cruzan_cruce5 = dataframe1_output_cruzan_cruce5
    df_name_dataframe2_output_nocruzan_cruce5 = dataframe2_output_nocruzan_cruce5
    df_name_dataframe1_output_cruzan_cruce6 = dataframe1_output_cruzan_cruce6
    df_name_dataframe2_output_nocruzan_cruce6 = dataframe2_output_nocruzan_cruce6

    df_name_Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif
    df_name_Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre
    df_name_Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif
    df_name_Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre

    df_name_dataframe_output_unificacion1 = dataframe_output_unificacion1
    df_name_dataframe_output_unificacion2 = dataframe_output_unificacion2
    df_name_dataframe_output_unificacion3 = dataframe_output_unificacion3
    df_name_dataframe_output_unificacion4 = dataframe_output_unificacion4

    df_name_dataframe_output_unificacion5 = dataframe_output_unificacion5
    df_name_dataframe_output_unificacion6 = dataframe_output_unificacion6
    
    
    #Creamos una función ventana que utilizaremos a lo largo del proceso
    #Cambio revisar
    Window_max_distance = Window.partitionBy('source_names')

    print('Cruce por NIF con maestro')
    #Calculamos el Id máximo que tenemos actualmente
    Max_Id = Dataframe_maestro.withColumn('Id', Dataframe_maestro['Id'].cast(IntegerType())).groupby().max('Id').collect()[0][0]


    #Analizamos que entidades del fichero de unidades organizativas hace match pro Nif con el maestro
    inner = Dataframe_uo.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').filter((F.col('CIF').isNotNull()) & (F.col('CIF') != ' ')).join(Dataframe_maestro.filter((F.col('CIF').isNotNull()) & (F.col('CIF') != ' ')).select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match'),
                                                               Dataframe_uo.CIF == Dataframe_maestro.CIF,
                                                               'inner').drop(Dataframe_maestro.CIF).drop_duplicates().withColumn('Inner', F.lit(1)).persist()


    dataframe2_output_nocruzan_cruce1 = Dataframe_uo.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNull()).drop('inner').drop('Entidad_Match').drop('Provincia_Match').dropDuplicates()
    dataframe1_output_cruzan_cruce1 = Dataframe_uo.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNotNull()).drop('inner').withColumn('Match', F.lit('NIF MCIN')).dropDuplicates()


    dataframe1_output_cruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1')
    dataframe1_output_cruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1')
    dataframe1_output_cruzan_cruce1.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe1_output_cruzan_cruce1' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)



    dataframe2_output_nocruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1')
    dataframe2_output_nocruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1')
    dataframe2_output_nocruzan_cruce1.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe2_output_nocruzan_cruce1' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    print('Cruza por NIF con maestro - Registros: ',dataframe1_output_cruzan_cruce1.count())
    print('No cruza por NIF con maestro - Registros: ',dataframe2_output_nocruzan_cruce1.count())
    #CIF es de ambos registros, Entidad_Norm y provincia_entidad son de uo y entidad_match y provincia match del maestro, match es el tipo de math

    # Archivo de error. Contiene las ayudas nacionales cuya entidad participante es pública y que han
    #  cruzado con el maestro por NIF, pero el nombre de la entidad participante y el nombre de la entidad en
    #  el maestro no cumplen los criterios de similitud por nombre, indicando que por nombre son entidades diferentes
    if(Bloque == '4.2.2.2'):
        df_casos_encontrados = dataframe1_output_cruzan_cruce1.select('CIF', 'Entidad_Norm','Entidad_Match','Provincia_Entidad','Provincia_Match')
        df_casos_encontrados = df_casos_encontrados.withColumn('final_score', udf_Distance_ratcliff_obershelp(F.col('Entidad_Norm'), F.col('Entidad_Match')))
        df_cruce_error = df_casos_encontrados.where(F.col("final_score")<Threshold)
        df_cruce_error.dropDuplicates().toPandas().to_csv(Ruta_Output + 'ERROR_ENTIDADES_MISMO_NIF_DISTINTO_NOMBRE' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
        print('Se crea fichero de error Se trata de información Europea ERROR_ENTIDADES_MISMO_NIF_DISTINTO_NOMBRE - Registros: ', df_cruce_error.count())
    
    print('Cruce por NIF con INVENTE')
    #Analizamos que entidades del fichero de unidades organizativas hace match pro Nif con invente
    inner = dataframe2_output_nocruzan_cruce1.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').join(Dataframe_invente.filter(F.col('CIF').isNotNull()).select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match'),
                                                               dataframe2_output_nocruzan_cruce1.CIF == Dataframe_invente.CIF,
                                                               'inner').drop(Dataframe_invente.CIF).drop_duplicates().withColumn('Inner', F.lit(1)).persist()

    dataframe2_output_nocruzan_cruce2 = dataframe2_output_nocruzan_cruce1.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNull()).drop('inner').drop('Entidad_Match').drop('Provincia_Match').dropDuplicates()
    dataframe1_output_cruzan_cruce2 = dataframe2_output_nocruzan_cruce1.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNotNull()).drop('inner').withColumn('Match', F.lit('NIF INVENTE')).dropDuplicates()

    dataframe1_output_cruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2')
    dataframe1_output_cruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2')
    dataframe1_output_cruzan_cruce2.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe1_output_cruzan_cruce2' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    dataframe2_output_nocruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2')
    dataframe2_output_nocruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2')
    dataframe2_output_nocruzan_cruce2.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe2_output_nocruzan_cruce2' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    print('Cruza por NIF con INVENTE - Registros: ',dataframe1_output_cruzan_cruce2.count())
    print('No Cruza por NIF con INVENTE - Registros: ',dataframe2_output_nocruzan_cruce2.count())



    #Cambio IMPORTANTE REVISAR 108 + 2063

    #Calculamos el fichero de invente con los cuales se obtuvo algún match
    Dataframe_nueva_invente_nif = dataframe1_output_cruzan_cruce2.select('CIF').join(Dataframe_invente.filter(F.col('CIF').isNotNull()),
                                                                                    dataframe1_output_cruzan_cruce2.CIF == Dataframe_invente.CIF,
                                                                                    'inner').drop(dataframe1_output_cruzan_cruce2.CIF).dropDuplicates()

    Dataframe_nueva_invente_nif.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_invente_nif')
    Dataframe_nueva_invente_nif = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_invente_nif')
    Dataframe_nueva_invente_nif.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Dataframe_nueva_invente_nif' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.join(Dataframe_maestro.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumn('inner', F.lit(1)),
                                                                  (Dataframe_nueva_invente_nif.CIF == Dataframe_maestro.CIF) &
                                                                  (Dataframe_nueva_invente_nif.Entidad_Norm == Dataframe_maestro.Entidad_Norm) &
                                                                  (Dataframe_nueva_invente_nif.Provincia_Entidad == Dataframe_maestro.Provincia_Entidad),
                                                                  'left').filter(F.col('inner').isNull()).drop('inner').drop(Dataframe_maestro.CIF).drop(Dataframe_maestro.Entidad_Norm).drop(Dataframe_maestro.Provincia_Entidad).withColumn('Id', row_number().over(Window.orderBy(monotonically_increasing_id())) + Max_Id)



    #Unificamos las nuevas entidades de invente en el maestro
    dataframe_output_unificacion1 = Dataframe_maestro.union(Dataframe_nueva_invente_nif.select(Dataframe_maestro.columns))


    dataframe_output_unificacion1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion1')
    dataframe_output_unificacion1 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion1')
    dataframe_output_unificacion1.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe_output_unificacion1' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    Max_Id = dataframe_output_unificacion1.withColumn('Id', dataframe_output_unificacion1['Id'].cast(IntegerType())).groupby().max('Id').collect()[0][0]

    print('Se inclyen en el maestro - Registros: ',Dataframe_nueva_invente_nif.count())
    print('Maestro actual - Registros: ',dataframe_output_unificacion1.count())

    #Analizamos que entidades del fichero de unidades organizativas hace match por Nif con DIR3

    #inner = dataframe2_output_nocruzan_cruce2.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').join(Dataframe_DIR3.filter(F.col('CIF').isNotNull()).select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match'),
    #                                                           dataframe2_output_nocruzan_cruce2.CIF == Dataframe_DIR3.CIF,
    #                                                           'inner').drop(Dataframe_DIR3.CIF).drop_duplicates().withColumn('Inner', F.lit(1)).persist()

    #Analizamos que entidades del fichero de unidades organizativas hace match por Nif con DIR3
    print('Cruce por NIF con DIR3')
    inner = dataframe2_output_nocruzan_cruce2.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').join(DIR3_aux.filter(F.col('CIF').isNotNull()).select('CIF', 'Entidad_Norm', 'Provincia_Entidad', 'N_NIVEL_JERARQUICO').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match'),
                                                               dataframe2_output_nocruzan_cruce2.CIF == DIR3_aux.CIF,
                                                               'inner').drop(DIR3_aux.CIF).drop_duplicates().withColumn('Inner', F.lit(1)).persist()

    #Cambio revisar si tiene sentido eliminar este paso (no es relevante ya que agropamos por nif)
    Window_max_jerarquia = Window.partitionBy('CIF', 'Entidad_Norm', 'Provincia_Entidad')

    inner = inner.withColumn('min_NIVEL', F.min('N_NIVEL_JERARQUICO').over(Window_max_jerarquia)).where(F.col('N_NIVEL_JERARQUICO') == F.col('min_NIVEL')).drop('min_NIVEL').drop('N_NIVEL_JERARQUICO')


    dataframe2_output_nocruzan_cruce3 = dataframe2_output_nocruzan_cruce2.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNull()).drop('inner').drop('Entidad_Match').drop('Provincia_Match').dropDuplicates()
    dataframe1_output_cruzan_cruce3 = dataframe2_output_nocruzan_cruce2.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNotNull()).drop('inner').withColumn('Match', F.lit('NIF DIR3')).dropDuplicates()

    #Guardamos resultados
    dataframe1_output_cruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3')
    dataframe1_output_cruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3')
    dataframe1_output_cruzan_cruce3.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe1_output_cruzan_cruce3' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    #Guardamos resultados
    dataframe2_output_nocruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3')
    dataframe2_output_nocruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3')
    dataframe2_output_nocruzan_cruce3.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe2_output_nocruzan_cruce3' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    print('Cruza por NIF con DIR3 - Registros: ',dataframe1_output_cruzan_cruce3.count())
    print('No Cruza por NIF con DIR3 - Registros: ',dataframe2_output_nocruzan_cruce3.count())
    
    
    #Calculamos el fichero de DIR con los cuales se obtuvo algún match

    #calculamos el inner porque tenemos que tener en cuenta el nivel jerarquico
    #inner = dataframe2_output_nocruzan_cruce2.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').join(Dataframe_DIR3.filter(F.col('CIF').isNotNull()).select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match'),
    #                                                           dataframe2_output_nocruzan_cruce2.CIF == Dataframe_DIR3.CIF,
    #                                                           'inner').drop(Dataframe_DIR3.CIF).drop_duplicates().withColumn('Inner', F.lit(1)).persist()



    #Calculamos el fichero de DIR con los cuales se obtuvo algún match

    #calculamos el inner porque tenemos que tener en cuenta el nivel jerarquico
    inner = dataframe2_output_nocruzan_cruce2.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').join(DIR3_aux.filter(F.col('CIF').isNotNull()).select('CIF', 'Entidad_Norm', 'Provincia_Entidad', 'N_NIVEL_JERARQUICO').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match'),
                                                               dataframe2_output_nocruzan_cruce2.CIF == DIR3_aux.CIF,
                                                               'inner').drop(DIR3_aux.CIF).drop_duplicates().withColumn('Inner', F.lit(1)).persist()

    Dataframe_nueva_DIR3_nif = dataframe1_output_cruzan_cruce3.select('CIF').join(DIR3_aux,
                                                                                  dataframe1_output_cruzan_cruce3.CIF == DIR3_aux.CIF,
                                                                                  'inner').drop(dataframe1_output_cruzan_cruce3.CIF).dropDuplicates()

    #Aplicamos función ventana
    Window_max_jerarquia = Window.partitionBy('CIF')
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('min_NIVEL', F.min('N_NIVEL_JERARQUICO').over(Window_max_jerarquia)).where(F.col('N_NIVEL_JERARQUICO') == F.col('min_NIVEL')).drop('min_NIVEL').drop('N_NIVEL_JERARQUICO')


    #Guardamos resultados

    Dataframe_nueva_DIR3_nif.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_DIR3_nif')
    Dataframe_nueva_DIR3_nif = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_DIR3_nif')
    Dataframe_nueva_DIR3_nif.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Dataframe_nueva_DIR3_nif' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)




    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.join(dataframe_output_unificacion1.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumn('inner', F.lit(1)),
                                                                  (Dataframe_nueva_DIR3_nif.CIF == dataframe_output_unificacion1.CIF) &
                                                                  (Dataframe_nueva_DIR3_nif.Entidad_Norm == dataframe_output_unificacion1.Entidad_Norm) &
                                                                  (Dataframe_nueva_DIR3_nif.Provincia_Entidad == dataframe_output_unificacion1.Provincia_Entidad),
                                                                  'left').filter(F.col('inner').isNull()).drop('inner').drop(dataframe_output_unificacion1.CIF).drop(dataframe_output_unificacion1.Entidad_Norm).drop(dataframe_output_unificacion1.Provincia_Entidad).withColumn('Id', row_number().over(Window.orderBy(monotonically_increasing_id())) + Max_Id)


    #Unificamos las nuevas entidades de DIR3 en el maestro
    dataframe_output_unificacion2 = dataframe_output_unificacion1.union(Dataframe_nueva_DIR3_nif.select(dataframe_output_unificacion1.columns))

    dataframe_output_unificacion2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion2')
    dataframe_output_unificacion2 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion2')
    dataframe_output_unificacion2.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe_output_unificacion2' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    Max_Id = dataframe_output_unificacion2.withColumn('Id', dataframe_output_unificacion2['Id'].cast(IntegerType())).groupby().max('Id').collect()[0][0]


    print('Se inclyen en el maestro - Registros: ',Dataframe_nueva_DIR3_nif.count())
    print('Maestro actual - Registros: ',dataframe_output_unificacion2.count())


    print('Cruzamos los registros que no han cruzado aun con el maestro')

    #Window_max_distance = Window.partitionBy('source_names', 'target_names')
    #aquí empieza el cruce por nombre con el fichero anterior de lo que no ha cruzado por nif con dir3 y el fichero maestro unificado dos que contiene las uniones de invente y dir3 por nif
    Match_Nombre_distance = calcular_Distance_ratcliff_obershelp_ElasticSearch_Ubicacion(spark,
                                                                                         syn_cities,
                                                                                         cities,
                                                                                         dataframe2_output_nocruzan_cruce3.select('Entidad_Norm', 'Provincia_Entidad'),
                                                                                         'Entidad_Norm',
                                                                                         'Provincia_Entidad',
                                                                                         None,
                                                                                         None,
                                                                                         dataframe_output_unificacion2.select('Entidad_Norm', 'Provincia_Entidad'),
                                                                                         'Entidad_Norm',
                                                                                         'Provincia_Entidad',
                                                                                         None,
                                                                                         None,
                                                                                         'indice',
                                                                                         0.93283582,
                                                                                         0.06716418,
                                                                                         None,
                                                                                         Threshold,
                                                                                         50)
    
    
    Match_Nombre_distance.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Match_Nombre_distance_Maestro' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    Match_Nombre_distance = Match_Nombre_distance.withColumn('final_score', F.when(F.col('source_names') != (F.col('target_names')), F.col('final_score')).otherwise(F.lit(1))).filter(F.col('final_score') > Threshold).withColumn('maxfinal_score', F.max('final_score').over(Window_max_distance)).where(F.col('final_score') == F.col('maxfinal_score')).drop('maxfinal_score')


    source = Match_Nombre_distance.select('source_names', 'source_municipality', 'target_names', 'target_municipality', 'final_score').withColumnRenamed('source_names',"Entidad_Norm").withColumnRenamed('source_municipality',"Provincia_Entidad").withColumnRenamed('target_names',"Entidad_Match").withColumnRenamed('target_municipality',"Provincia_Match")

    inner = dataframe2_output_nocruzan_cruce3.join(source,
                                                  (dataframe2_output_nocruzan_cruce3.Entidad_Norm == source.Entidad_Norm) &
                                                  (dataframe2_output_nocruzan_cruce3.Provincia_Entidad == source.Provincia_Entidad),
                                                  'inner').drop(source.Entidad_Norm).drop(source.Provincia_Entidad).withColumn('Inner', F.lit(1)).persist()

    dataframe2_output_nocruzan_cruce4 = dataframe2_output_nocruzan_cruce3.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNull()).drop('inner').drop('Entidad_Match').drop('Provincia_Match').drop('final_score').dropDuplicates()
    dataframe1_output_cruzan_cruce4 = dataframe2_output_nocruzan_cruce3.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNotNull()).drop('inner').withColumn('Match', F.lit('NOMBRE MCIN')).dropDuplicates()


 
    dataframe1_output_cruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4')
    dataframe1_output_cruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4')
    dataframe1_output_cruzan_cruce4.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe1_output_cruzan_cruce4' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)



    dataframe2_output_nocruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4')
    dataframe2_output_nocruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4')
    dataframe2_output_nocruzan_cruce4.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe2_output_nocruzan_cruce4' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    print('Cruza por por nombre con maestro: ',dataframe1_output_cruzan_cruce4.count())
    print('No Cruza por por nombre con maestro - Registros: ',dataframe2_output_nocruzan_cruce4.count())
    


    #aquí se cruza con nombre con invente. #PUBLICA

    if (Dataframe_invente.count() == 0):
        schema = StructType([
        StructField('source_names', StringType(), True),
        StructField('target_names', StringType(), True),
        StructField('source_municipality', StringType(), True),
        StructField('target_municipality', StringType(), True),
        StructField('source_city', StringType(), True),
        StructField('target_city', StringType(), True),
        StructField('source_country', StringType(), True),
        StructField('target_country', StringType(), True),
        StructField('name_score', StringType(), True),
        StructField('city_score', StringType(), True),
        StructField('mun_score', StringType(), True),
        StructField('country_score', StringType(), True),
        StructField('weighted_name_score', StringType(), True),
        StructField('weighted_city_score', StringType(), True),
        StructField('weighted_mun_score', StringType(), True),
        StructField('weighted_country_score', StringType(), True),
        StructField('final_score', StringType(), True),
        StructField('source_names_stopwords', StringType(), True),
        StructField('target_names_stopwords', StringType(), True)])

        #Creamos el fichero Tabla_Entidades_id en dataframe de spark
        Match_Nombre_distance = spark.createDataFrame([], schema)
    else:
        print('Cruzamos por nombre con INVENTE')
        Match_Nombre_distance = calcular_Distance_ratcliff_obershelp_ElasticSearch_Ubicacion(spark,
                                                                                             syn_cities,
                                                                                             cities,
                                                                                             dataframe2_output_nocruzan_cruce4,
                                                                                             'Entidad_Norm',
                                                                                             'Provincia_Entidad',
                                                                                             None,
                                                                                             None,
                                                                                             Dataframe_invente,
                                                                                             'Entidad_Norm',
                                                                                             'Provincia_Entidad',
                                                                                             None,
                                                                                             None,
                                                                                             'indice',
                                                                                             0.93283582,
                                                                                             0.06716418,
                                                                                             None,
                                                                                             Threshold,
                                                                                             50)

    
    Match_Nombre_distance.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Match_Nombre_distance_Invente' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    Match_Nombre_distance = Match_Nombre_distance.withColumn('final_score', F.when(F.col('source_names') != (F.col('target_names')), F.col('final_score')).otherwise(F.lit(1))).filter(F.col('final_score') > Threshold).withColumn('maxfinal_score', F.max('final_score').over(Window_max_distance)).where(F.col('final_score') == F.col('maxfinal_score')).drop('maxfinal_score')



    source = Match_Nombre_distance.select('source_names', 'source_municipality', 'target_names', 'target_municipality', 'final_score').withColumnRenamed('source_names',"Entidad_Norm").withColumnRenamed('source_municipality',"Provincia_Entidad").withColumnRenamed('target_names',"Entidad_Match").withColumnRenamed('target_municipality',"Provincia_Match")

    inner = dataframe2_output_nocruzan_cruce4.join(source,
                                                  (dataframe2_output_nocruzan_cruce4.Entidad_Norm == source.Entidad_Norm) &
                                                  (dataframe2_output_nocruzan_cruce4.Provincia_Entidad == source.Provincia_Entidad),
                                                  'inner').drop(source.Entidad_Norm).drop(source.Provincia_Entidad).withColumn('Inner', F.lit(1)).persist()

    dataframe2_output_nocruzan_cruce5 = dataframe2_output_nocruzan_cruce4.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNull()).drop('inner').drop('Entidad_Match').drop('Provincia_Match').drop('final_score').dropDuplicates()
    dataframe1_output_cruzan_cruce5 = dataframe2_output_nocruzan_cruce4.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNotNull()).drop('inner').withColumn('Match', F.lit('NOMBRE INVENTE')).dropDuplicates()


    dataframe1_output_cruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5')
    dataframe1_output_cruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5')
    dataframe1_output_cruzan_cruce5.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe1_output_cruzan_cruce5' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)



    dataframe2_output_nocruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5')
    dataframe2_output_nocruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5')
    dataframe2_output_nocruzan_cruce5.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe2_output_nocruzan_cruce5' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    print('Cruza por por nombre con invente: ',dataframe1_output_cruzan_cruce5.count())
    print('No Cruza por por nombre con invente - Registros: ',dataframe2_output_nocruzan_cruce5.count())
    


    Dataframe_nueva_invente_nombre = dataframe1_output_cruzan_cruce5.select('Entidad_Match', 'Provincia_Match').join(Dataframe_invente,
                                                                                                                     (dataframe1_output_cruzan_cruce5.Entidad_Match == Dataframe_invente.Entidad_Norm) &
                                                                                                                     (dataframe1_output_cruzan_cruce5.Provincia_Match == Dataframe_invente.Provincia_Entidad),
                                                                                                                     'inner').drop(dataframe1_output_cruzan_cruce5.Entidad_Match).drop(dataframe1_output_cruzan_cruce5.Provincia_Match).dropDuplicates()


    Dataframe_nueva_invente_nombre.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_invente_nombre')
    Dataframe_nueva_invente_nombre = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_invente_nombre')
    Dataframe_nueva_invente_nombre.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Dataframe_nueva_invente_nombre' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.join(dataframe_output_unificacion2.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumn('inner', F.lit(1)),
                                                                  (Dataframe_nueva_invente_nombre.CIF == dataframe_output_unificacion2.CIF) &
                                                                  (Dataframe_nueva_invente_nombre.Entidad_Norm == dataframe_output_unificacion2.Entidad_Norm) &
                                                                  (Dataframe_nueva_invente_nombre.Provincia_Entidad == dataframe_output_unificacion2.Provincia_Entidad),
                                                                  'left').filter(F.col('inner').isNull()).drop('inner').drop(dataframe_output_unificacion2.CIF).drop(dataframe_output_unificacion2.Entidad_Norm).drop(dataframe_output_unificacion2.Provincia_Entidad).withColumn('Id', row_number().over(Window.orderBy(monotonically_increasing_id())) + Max_Id)



    #Unificamos las nuevas entidades de invente en el maestro
    dataframe_output_unificacion3 = dataframe_output_unificacion2.union(Dataframe_nueva_invente_nombre.select(dataframe_output_unificacion2.columns))


    dataframe_output_unificacion3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion3')
    dataframe_output_unificacion3 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion3')
    dataframe_output_unificacion3.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe_output_unificacion3' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    Max_Id = dataframe_output_unificacion3.withColumn('Id', dataframe_output_unificacion3['Id'].cast(IntegerType())).groupby().max('Id').collect()[0][0]

    print('Se inclyen en el maestro - Registros: ',Dataframe_nueva_invente_nombre.count())
    print('Maestro actual - Registros: ',dataframe_output_unificacion3.count())



    #Aquí cruzamos con DIR3 por nombre
    if (Dataframe_DIR3.count() == 0):
        schema = StructType([
        StructField('source_names', StringType(), True),
        StructField('target_names', StringType(), True),
        StructField('source_municipality', StringType(), True),
        StructField('target_municipality', StringType(), True),
        StructField('source_city', StringType(), True),
        StructField('target_city', StringType(), True),
        StructField('source_country', StringType(), True),
        StructField('target_country', StringType(), True),
        StructField('name_score', StringType(), True),
        StructField('city_score', StringType(), True),
        StructField('mun_score', StringType(), True),
        StructField('country_score', StringType(), True),
        StructField('weighted_name_score', StringType(), True),
        StructField('weighted_city_score', StringType(), True),
        StructField('weighted_mun_score', StringType(), True),
        StructField('weighted_country_score', StringType(), True),
        StructField('final_score', StringType(), True),
        StructField('source_names_stopwords', StringType(), True),
        StructField('target_names_stopwords', StringType(), True)])

        #Creamos el fichero Tabla_Entidades_id en dataframe de spark
        Match_Nombre_distance = spark.createDataFrame([], schema)
    else:
        print('Cruzamos por nombre con DIR3')
        Match_Nombre_distance = calcular_Distance_ratcliff_obershelp_ElasticSearch_Ubicacion(spark,
                                                                                             syn_cities,
                                                                                             cities,
                                                                                             dataframe2_output_nocruzan_cruce5,
                                                                                             'Entidad_Norm',
                                                                                             'Provincia_Entidad',
                                                                                             None,
                                                                                             None,
                                                                                             Dataframe_DIR3,
                                                                                             'Entidad_Norm',
                                                                                             'Provincia_Entidad',
                                                                                             None,
                                                                                             None,
                                                                                             'indice',
                                                                                             0.93283582,
                                                                                             0.06716418,
                                                                                             None,
                                                                                             Threshold,
                                                                                             50)
    
    Match_Nombre_distance.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Match_Nombre_distance_DIR3' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    Match_Nombre_distance = Match_Nombre_distance.withColumn('final_score', F.when(F.col('source_names') != (F.col('target_names')), F.col('final_score')).otherwise(F.lit(1))).filter(F.col('final_score') > Threshold).withColumn('maxfinal_score', F.max('final_score').over(Window_max_distance)).where(F.col('final_score') == F.col('maxfinal_score')).drop('maxfinal_score').filter((F.col('source_municipality') == F.col('target_municipality')) | (F.col('source_municipality') == '77') | (F.col('source_municipality') == '88') | (F.col('source_municipality').isNull()) | (F.col('target_municipality') == '77') | (F.col('target_municipality') == '88') | (F.col('target_municipality').isNull()))
    
    
    Match_Nombre_distance.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Match_Nombre_distance_DIR3_depurado' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)



    source = Match_Nombre_distance.select('source_names', 'source_municipality', 'target_names', 'target_municipality', 'final_score').withColumnRenamed('source_names',"Entidad_Norm").withColumnRenamed('source_municipality',"Provincia_Entidad").withColumnRenamed('target_names',"Entidad_Match").withColumnRenamed('target_municipality',"Provincia_Match")

    inner = dataframe2_output_nocruzan_cruce5.join(source,
                                                  (dataframe2_output_nocruzan_cruce5.Entidad_Norm == source.Entidad_Norm) &
                                                  (dataframe2_output_nocruzan_cruce5.Provincia_Entidad == source.Provincia_Entidad),
                                                  'inner').drop(source.Entidad_Norm).drop(source.Provincia_Entidad).withColumn('Inner', F.lit(1)).persist()
    
    
    inner = inner.join(DIR3_aux.select('Entidad_Norm', 'Provincia_Entidad', 'N_NIVEL_JERARQUICO'),
                  (inner.Entidad_Match == DIR3_aux.Entidad_Norm) &
                  (inner.Provincia_Match == DIR3_aux.Provincia_Entidad),
                  'left').drop(DIR3_aux.Entidad_Norm).drop(DIR3_aux.Provincia_Entidad)

    inner.dropDuplicates().toPandas().to_csv(Ruta_Output + 'inner_DIR3_NOMBRE' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    #Aplicamos función ventana
    Window_max_jerarquia = Window.partitionBy('CIF', 'Entidad_Norm', 'Provincia_Entidad')
    inner = inner.withColumn('min_NIVEL', F.min('N_NIVEL_JERARQUICO').over(Window_max_jerarquia)).where(F.col('N_NIVEL_JERARQUICO') == F.col('min_NIVEL')).drop('min_NIVEL').drop('N_NIVEL_JERARQUICO')


    dataframe2_output_nocruzan_cruce6 = dataframe2_output_nocruzan_cruce5.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNull()).drop('inner').drop('Entidad_Match').drop('Provincia_Match').drop('final_score').dropDuplicates()
    dataframe1_output_cruzan_cruce6 = dataframe2_output_nocruzan_cruce5.join(inner, ['CIF', 'Entidad_Norm', 'Provincia_Entidad'], 'left').where(F.col('inner').isNotNull()).drop('inner').withColumn('Match', F.lit('NOMBRE DIR3')).dropDuplicates()

    dataframe1_output_cruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6')
    dataframe1_output_cruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6')
    dataframe1_output_cruzan_cruce6.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe1_output_cruzan_cruce6' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    dataframe2_output_nocruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6')
    dataframe2_output_nocruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6')
    dataframe2_output_nocruzan_cruce6.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe2_output_nocruzan_cruce6' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    print('Cruza por por nombre con DIR3: ',dataframe1_output_cruzan_cruce6.count())
    print('No Cruza por por nombre con DIR3 - Registros: ',dataframe2_output_nocruzan_cruce6.count())



    #Creamos el fichero Dataframe_nueva_DIR3_nombre
    Dataframe_nueva_DIR3_nombre = dataframe1_output_cruzan_cruce6.select('Entidad_Match', 'Provincia_Match').join(DIR3_aux.select('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'N_NIVEL_JERARQUICO'),
                                                                                                                  (dataframe1_output_cruzan_cruce6.Entidad_Match == DIR3_aux.Entidad_Norm) &
                                                                                                                  (dataframe1_output_cruzan_cruce6.Provincia_Match == DIR3_aux.Provincia_Entidad),
                                                                                                                  'inner').drop(dataframe1_output_cruzan_cruce6.Entidad_Match).drop(dataframe1_output_cruzan_cruce6.Provincia_Match).dropDuplicates()

    
    
    Dataframe_nueva_DIR3_nombre.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Dataframe_nueva_DIR3_nombre_N_NIVEL_JERARQUICO' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    #Aplicamos función ventana
    Window_max_jerarquia = Window.partitionBy('Entidad_Norm', 'Provincia_Entidad')
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('min_NIVEL', F.min('N_NIVEL_JERARQUICO').over(Window_max_jerarquia)).where(F.col('N_NIVEL_JERARQUICO') == F.col('min_NIVEL')).drop('min_NIVEL').drop('N_NIVEL_JERARQUICO')



    #Guardamos el fichero Dataframe_nueva_DIR3_nombre
    Dataframe_nueva_DIR3_nombre.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_Dataframe_nueva_DIR3_nombre)
    Dataframe_nueva_DIR3_nombre = spark.read.parquet(Ruta_Output + df_name_Dataframe_nueva_DIR3_nombre)
    Dataframe_nueva_DIR3_nombre.dropDuplicates().toPandas().to_csv(Ruta_Output + df_name_Dataframe_nueva_DIR3_nombre + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.join(dataframe_output_unificacion3.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumn('inner', F.lit(1)),
                                                                  (Dataframe_nueva_DIR3_nombre.CIF == dataframe_output_unificacion3.CIF) &
                                                                  (Dataframe_nueva_DIR3_nombre.Entidad_Norm == dataframe_output_unificacion3.Entidad_Norm) &
                                                                  (Dataframe_nueva_DIR3_nombre.Provincia_Entidad == dataframe_output_unificacion3.Provincia_Entidad),
                                                                  'left').filter(F.col('inner').isNull()).drop('inner').drop(dataframe_output_unificacion3.CIF).drop(dataframe_output_unificacion3.Entidad_Norm).drop(dataframe_output_unificacion3.Provincia_Entidad).withColumn('Id', row_number().over(Window.orderBy(monotonically_increasing_id())) + Max_Id)

    #Unificamos las nuevas entidades de DIR3 en el maestro
    dataframe_output_unificacion4 = dataframe_output_unificacion3.union(Dataframe_nueva_DIR3_nombre.select(dataframe_output_unificacion3.columns))

    dataframe_output_unificacion4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion4')
    dataframe_output_unificacion4 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion4')
    dataframe_output_unificacion4.dropDuplicates().toPandas().to_csv(Ruta_Output + 'dataframe_output_unificacion4' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    Max_Id = dataframe_output_unificacion4.withColumn('Id', dataframe_output_unificacion4['Id'].cast(IntegerType())).groupby().max('Id').collect()[0][0]

    #FIN PUBLICA

    print('Se inclyen en el maestro - Registros: ',Dataframe_nueva_DIR3_nombre.count())
    print('Maestro actual - Registros: ',dataframe_output_unificacion4.count())
    
    
    #Guardamos copia de seguridad de todos los ficheros
    dataframe1_output_cruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1_pre_cs')
    dataframe2_output_nocruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1_pre_cs')
    dataframe1_output_cruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2_pre_cs')
    dataframe2_output_nocruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2_pre_cs')
    dataframe1_output_cruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3_pre_cs')
    dataframe2_output_nocruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3_pre_cs')
    dataframe1_output_cruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4_pre_cs')
    dataframe2_output_nocruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4_pre_cs')
    dataframe1_output_cruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5_pre_cs')
    dataframe2_output_nocruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5_pre_cs')
    dataframe1_output_cruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6_pre_cs')
    dataframe2_output_nocruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6_pre_cs')
    dataframe_output_unificacion1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion1_pre_cs')
    dataframe_output_unificacion2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion2_pre_cs')
    dataframe_output_unificacion3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion3_pre_cs')
    dataframe_output_unificacion4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion4_pre_cs')
    #Leemos copia de seguridad de todos los ficheros
    dataframe1_output_cruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1_pre_cs')
    dataframe2_output_nocruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1_pre_cs')
    dataframe1_output_cruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2_pre_cs')
    dataframe2_output_nocruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2_pre_cs')
    dataframe1_output_cruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3_pre_cs')
    dataframe2_output_nocruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3_pre_cs')
    dataframe1_output_cruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4_pre_cs')
    dataframe2_output_nocruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4_pre_cs')
    dataframe1_output_cruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5_pre_cs')
    dataframe2_output_nocruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5_pre_cs')
    dataframe1_output_cruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6_pre_cs')
    dataframe2_output_nocruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6_pre_cs')

    dataframe_output_unificacion1 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion1_pre_cs')
    dataframe_output_unificacion2 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion2_pre_cs')
    dataframe_output_unificacion3 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion3_pre_cs')
    dataframe_output_unificacion4 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion4_pre_cs')    
  


    #La unificación 5 en este punto es la unificación 4, la creamos
    dataframe_output_unificacion5 = dataframe_output_unificacion4

    #Guardamos resultados

    dataframe_output_unificacion5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'df_name_dataframe_output_unificacion5')
    dataframe_output_unificacion5 = spark.read.parquet(Ruta_Output + 'df_name_dataframe_output_unificacion5')
    dataframe_output_unificacion5.dropDuplicates().toPandas().to_csv(Ruta_Output + 'df_name_dataframe_output_unificacion5' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    #Mostramos resultados
    print(dataframe_output_unificacion5.count())
    dataframe_output_unificacion5.show()


    print('Maestro actual - Registros????5: ',dataframe_output_unificacion5.count())
    #lo quito, creo que no tiene sentido

    #Creamos dataframe_output_unificacion5, la union de todo lo qeu hizo match
    dataframe_output_unificacion6 = dataframe1_output_cruzan_cruce1.select(dataframe1_output_cruzan_cruce1.columns) .union(dataframe1_output_cruzan_cruce2.select(dataframe1_output_cruzan_cruce1.columns)) .union(dataframe1_output_cruzan_cruce3.select(dataframe1_output_cruzan_cruce1.columns)) .union(dataframe1_output_cruzan_cruce4.select(dataframe1_output_cruzan_cruce1.columns)) .union(dataframe1_output_cruzan_cruce5.select(dataframe1_output_cruzan_cruce1.columns)) .union(dataframe1_output_cruzan_cruce6.select(dataframe1_output_cruzan_cruce1.columns)) .union(dataframe2_output_nocruzan_cruce6.withColumn('Id', F.lit(None).cast(StringType())).withColumn('Entidad_Match', F.lit(None).cast(StringType())).withColumn('Provincia_Match', F.lit(None).cast(StringType())).withColumn('Match', F.lit(None).cast(StringType())).select(dataframe1_output_cruzan_cruce1.columns))


    print('Maestro actual - Registros????6: ',dataframe_output_unificacion6.count())

    #Mostramos los distintos tipos de match que tenemos en el fichero, los nulos son los que no han hecho match
    print(dataframe_output_unificacion6.groupBy('Match').count().groupBy('Match').count().count())
    dataframe_output_unificacion6.groupBy('Match').count().groupBy('Match').count().show(100, False)

    
    #Separamos los ficheros en tres partes, los que hicieron match pro nombre, por nif y el resto
    dataframe_output_unificacion6_Nifs = dataframe_output_unificacion6.filter((F.col('Match').contains('NIF'))).dropDuplicates()
    dataframe_output_unificacion6_Nombres = dataframe_output_unificacion6.filter((F.col('Match').contains('NOMBRE'))).dropDuplicates()
    dataframe_output_unificacion6_Resto = dataframe_output_unificacion6.filter(F.col('Match').isNull()).dropDuplicates()


    
    
    #Buscamos similares por NIF
    dataframe_output_unificacion6_Nifs = dataframe_output_unificacion6_Nifs.filter((F.col('CIF').isNotNull()) | (F.col('CIF') != ' ')).join(dataframe_output_unificacion4.filter((F.col('CIF').isNotNull()) | (F.col('CIF') != ' ')),
                                                                                 (dataframe_output_unificacion6_Nifs.Entidad_Match == dataframe_output_unificacion4.Entidad_Norm) &
                                                                                 (dataframe_output_unificacion6_Nifs.Provincia_Match == dataframe_output_unificacion4.Provincia_Entidad) &
                                                                                 (dataframe_output_unificacion6_Nifs.CIF == dataframe_output_unificacion4.CIF),
                                                                                 'left').drop(dataframe_output_unificacion4.CIF).drop(dataframe_output_unificacion4.Entidad_Norm).drop(dataframe_output_unificacion4.Provincia_Entidad)
    #Buscamos similares por Nombre
    dataframe_output_unificacion6_Nombres = dataframe_output_unificacion6_Nombres.join(dataframe_output_unificacion4,
                                                                                       (dataframe_output_unificacion6_Nombres.Entidad_Match == dataframe_output_unificacion4.Entidad_Norm) &
                                                                                       (dataframe_output_unificacion6_Nombres.Provincia_Match == dataframe_output_unificacion4.Provincia_Entidad),
                                                                                       'left').drop(dataframe_output_unificacion4.CIF).drop(dataframe_output_unificacion4.Entidad_Norm).drop(dataframe_output_unificacion4.Provincia_Entidad)
    #obtenemos los que siguen sin Id
    dataframe_output_unificacion6_Resto = dataframe_output_unificacion6_Resto.withColumn('Id', F.lit(None).cast(StringType()))

    #Unimos los 3 casos anteriores
    dataframe_output_unificacion6 = dataframe_output_unificacion6_Nifs.union(dataframe_output_unificacion6_Nombres.select(dataframe_output_unificacion6_Nifs.columns)).union(dataframe_output_unificacion6_Resto.select(dataframe_output_unificacion6_Nifs.columns)).dropDuplicates()

    #Guardamos el fichero
    print('Guardamos dataframe_output_unificacion6')
    dataframe_output_unificacion6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'df_name_dataframe_output_unificacion6')
    dataframe_output_unificacion6 = spark.read.parquet(Ruta_Output + 'df_name_dataframe_output_unificacion6')
    dataframe_output_unificacion6.dropDuplicates().toPandas().to_csv(Ruta_Output + 'df_name_dataframe_output_unificacion6' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)


    print(dataframe_output_unificacion6.count())
    dataframe_output_unificacion6.show()

    #En este paso vamos a intentar hacer matchs por nombre entre los registros sin Id vs los que si tienen Id,
    #hemos visto que esto no funciona al 100% en la primera vuelta, por lo que se ha creadeo un bucle While,
    #en el cual mientras se haga al menos un match en esa vuelta, 
    #se repetira el proceso al menos una vez más, así hasta conseguir 0 matchs

    vuelta = 0
    Match_Nombre_distance_count = 1

    while(Match_Nombre_distance_count != 0):

        vuelta += 1
        print('Vuelta: ', vuelta)
        
        dataframe_output_unificacion6_id = dataframe_output_unificacion6.filter(F.col('Id').isNotNull())
        dataframe_output_unificacion6_no_id = dataframe_output_unificacion6.filter(F.col('Id').isNull())


        dataframe_output_unificacion6_id_NIF = dataframe_output_unificacion6_id.select('Id', 'CIF', 'Entidad_Norm', 'Provincia_Entidad').withColumnRenamed('CIF', 'CIF_Match').withColumnRenamed('Entidad_Norm', 'Entidad_Match').withColumnRenamed('Provincia_Entidad', 'Provincia_Match').withColumn('Match', F.lit('Id Match'))


        dataframe_output_unificacion6_no_id = dataframe_output_unificacion6_no_id.select('CIF', 'Entidad_Norm', 'Provincia_Entidad').join(dataframe_output_unificacion6_id_NIF.filter((F.col('CIF').isNotNull()) & (F.col('CIF') != ' ')),
                                                                                                                dataframe_output_unificacion6_no_id.CIF == dataframe_output_unificacion6_id_NIF.CIF_Match,
                                                                                                                'left').drop(dataframe_output_unificacion6_id_NIF.CIF_Match)

        dataframe_output_unificacion6 = dataframe_output_unificacion6_id.union(dataframe_output_unificacion6_no_id.select(dataframe_output_unificacion6_id.columns))

        Match_NIF_count = dataframe_output_unificacion6_no_id.filter(F.col('Id').isNotNull()).count()
        print('Se han encontrado ', Match_NIF_count, ' relaciones por NIF')

        dataframe_output_unificacion6_id = dataframe_output_unificacion6.filter(F.col('Id').isNotNull())
        dataframe_output_unificacion6_no_id = dataframe_output_unificacion6.filter(F.col('Id').isNull())

        print('Ejecutamos elastic Id vs no Id')
        
        schema = StructType([
            StructField('source_names', StringType(), True),
            StructField('target_names', StringType(), True),
            StructField('source_municipality', StringType(), True),
            StructField('target_municipality', StringType(), True),
            StructField('source_city', StringType(), True),
            StructField('target_city', StringType(), True),
            StructField('source_country', StringType(), True),
            StructField('target_country', StringType(), True),
            StructField('name_score', StringType(), True),
            StructField('city_score', StringType(), True),
            StructField('mun_score', StringType(), True),
            StructField('country_score', StringType(), True),
            StructField('weighted_name_score', StringType(), True),
            StructField('weighted_city_score', StringType(), True),
            StructField('weighted_mun_score', StringType(), True),
            StructField('weighted_country_score', StringType(), True),
            StructField('final_score', StringType(), True),
            StructField('source_names_stopwords', StringType(), True),
            StructField('target_names_stopwords', StringType(), True)])

        #Creamos el fichero Tabla_Entidades_id en dataframe de spark
        Match_Nombre_distance = spark.createDataFrame([], schema)
        
        #CAMBIO IMPORTANTE QUITAMOS LOS MATCHS POR NOMBRE
        #para recuperar los matchs por nombre, comentar la linea de debajo
        Match_Nombre_distance = Match_Nombre_distance.where(F.lit(1)==F.lit(0)).dropDuplicates()

        Match_Nombre_distance_count = Match_Nombre_distance.count()
        print('Se han encontrado ', Match_Nombre_distance_count, ' relaciones por nombre')

        Match_Nombre_distance.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Match_Nombre_distance_Id_VS_noId_' + str(vuelta) + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

        #Cruzamos los registros sin id con el resultado del calculo de distancias por nombre con los que si tenian Id para buscar algunos que hagan match y asignarles el Id
        dataframe_output_unificacion6_no_id = dataframe_output_unificacion6_no_id.join(Match_Nombre_distance.select('source_names', 'source_municipality', 'target_names', 'target_municipality'),
                                                                                       (dataframe_output_unificacion6_no_id.Entidad_Norm == Match_Nombre_distance.source_names) &
                                                                                       (dataframe_output_unificacion6_no_id.Provincia_Entidad == Match_Nombre_distance.source_municipality),
                                                                                       'left').drop(Match_Nombre_distance.source_names).drop(Match_Nombre_distance.source_municipality)

        #Creamos un fichero con las columnas que necesitamos
        dataframe_output_unificacion6_id_aux = dataframe_output_unificacion6_id.select('Entidad_Norm', 'Provincia_Entidad', 'Id').withColumn('Entidad_Match', F.col('Entidad_Norm')).withColumn('Provincia_Match', F.col('Provincia_Entidad')).withColumn('Match', F.lit('Id Match')).withColumnRenamed('Entidad_Norm', 'target_names').withColumnRenamed('Provincia_Entidad', 'target_municipality')

        #Intentamos añadir el Id a los registros
        dataframe_output_unificacion6_no_id = dataframe_output_unificacion6_no_id.drop('Id').join(dataframe_output_unificacion6_id_aux,
                                                                                                  (dataframe_output_unificacion6_no_id.target_names == dataframe_output_unificacion6_id_aux.target_names) &
                                                                                                  (dataframe_output_unificacion6_no_id.target_municipality == dataframe_output_unificacion6_id_aux.target_municipality),
                                                                                                  'left').drop(dataframe_output_unificacion6_no_id.target_names).drop(dataframe_output_unificacion6_no_id.target_municipality).drop(dataframe_output_unificacion6_no_id.Entidad_Match).drop(dataframe_output_unificacion6_no_id.Provincia_Match).drop(dataframe_output_unificacion6_no_id.Match).drop(dataframe_output_unificacion6_id_aux.target_names).drop(dataframe_output_unificacion6_id_aux.target_municipality)

        #Unimos los ficheros que no tenian ID antes del proceso con los que si tenian
        dataframe_output_unificacion6 = dataframe_output_unificacion6_id.union(dataframe_output_unificacion6_no_id.select(dataframe_output_unificacion6_id.columns))

        print('Guardamos dataframe_output_unificacion6')
        dataframe_output_unificacion6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'df_name_dataframe_output_unificacion6_2_' + str(vuelta))
        dataframe_output_unificacion6 = spark.read.parquet(Ruta_Output + 'df_name_dataframe_output_unificacion6_2_' + str(vuelta))
        dataframe_output_unificacion6.dropDuplicates().toPandas().to_csv(Ruta_Output + 'df_name_dataframe_output_unificacion6_2_' + str(vuelta) + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    #Los registros que no han cruzado con nada serán los registros que se den de alta en el maestro, aunque primero tenemos que buscar similares entre ellos que será el proceso realizado a continuación
    altas_nuevas = dataframe_output_unificacion6.filter(F.col('Id').isNull()).drop('Id')

    altas_nuevas.show()

    altas_nuevas.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'altas_nuevas')
    
    altas_nuevas.dropDuplicates().toPandas().to_csv(Ruta_Output + 'altas_nuevas' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    #De los registros que no tenemos con Id, buscamos similares consigo mismos utilizando ElasticSearch
    #distance_altas_nuevas = calcular_Distance_ratcliff_obershelp_ElasticSearch_Ubicacion(spark,
    #                                                                                     syn_cities,
    #                                                                                     cities,
    #                                                                                     altas_nuevas,
    #                                                                                     'Entidad_Norm',
    #                                                                                     'Provincia_Entidad',
    #                                                                                     None,
    #                                                                                     None,
    #                                                                                     altas_nuevas,
    #                                                                                     'Entidad_Norm',
    #                                                                                     'Provincia_Entidad',
    #                                                                                     None,
    #                                                                                     None,
    #                                                                                     'indice',
    #                                                                                     0.93283582,
    #                                                                                     0.06716418,
    #                                                                                     None,
    #                                                                                     Threshold,
    #                                                                                     50).withColumn('final_score', F.when(F.col('source_names') != (F.col('target_names')), F.col('final_score')).otherwise(F.lit(1))).filter(F.col('final_score') > Threshold)
    
    schema = StructType([
        StructField('source_names', StringType(), True),
        StructField('target_names', StringType(), True),
        StructField('source_municipality', StringType(), True),
        StructField('target_municipality', StringType(), True),
        StructField('source_city', StringType(), True),
        StructField('target_city', StringType(), True),
        StructField('source_country', StringType(), True),
        StructField('target_country', StringType(), True),
        StructField('name_score', StringType(), True),
        StructField('city_score', StringType(), True),
        StructField('mun_score', StringType(), True),
        StructField('country_score', StringType(), True),
        StructField('weighted_name_score', StringType(), True),
        StructField('weighted_city_score', StringType(), True),
        StructField('weighted_mun_score', StringType(), True),
        StructField('weighted_country_score', StringType(), True),
        StructField('final_score', StringType(), True),
        StructField('source_names_stopwords', StringType(), True),
        StructField('target_names_stopwords', StringType(), True)])

    #Creamos el fichero Tabla_Entidades_id en dataframe de spark
    distance_altas_nuevas = spark.createDataFrame([], schema)
    
    #CAMBIO IMPORTANTE QUITAMOS LOS MATCHS POR NOMBRE
    #para recuperar los matchs por nombre, comentar la linea de debajo
    distance_altas_nuevas = distance_altas_nuevas.where(F.lit(1)==F.lit(0)).dropDuplicates()

    distance_altas_nuevas.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Match_Nombre_distance_noId' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    distance_altas_nuevas = distance_altas_nuevas.withColumn('source_municipality',distance_altas_nuevas.source_municipality.cast(StringType()))
    distance_altas_nuevas = distance_altas_nuevas.withColumn('target_municipality',distance_altas_nuevas.source_municipality.cast(StringType()))

    altas_nuevas = altas_nuevas.withColumn('Provincia_Entidad',altas_nuevas.Provincia_Entidad.cast(StringType()))

    pd_distance_altas_nuevas = distance_altas_nuevas.toPandas()
    pd_altas_nuevas = altas_nuevas.toPandas()


    #pd_distance_altas_nuevas = pd_distance_altas_nuevas[(pd_distance_altas_nuevas['source_names'] != pd_distance_altas_nuevas['target_names']) |
    #                                                    (pd_distance_altas_nuevas['source_municipality'] != pd_distance_altas_nuevas['target_municipality'])]

    pd_distance_altas_nuevas = pd_distance_altas_nuevas[['source_names', 'source_municipality', 'target_names', 'target_municipality']]

    source_pd_altas_nuevas = pd_altas_nuevas[['Entidad_Norm', 'CIF', 'Provincia_Entidad']]
    source_pd_altas_nuevas.columns = ['source_names', 'source_CIF', 'source_municipality']

    target_pd_altas_nuevas = pd_altas_nuevas[['Entidad_Norm', 'CIF', 'Provincia_Entidad']]
    target_pd_altas_nuevas.columns = ['target_names', 'target_CIF', 'target_municipality']
    


    pd_distance_altas_nuevas = pd_distance_altas_nuevas.merge(source_pd_altas_nuevas,
                                                              how='left',
                                                              on=['source_names', 'source_municipality']).merge(target_pd_altas_nuevas,
                                                                                                                how='left',
                                                                                                                on=['target_names', 'target_municipality'])


    pd_distance_altas_nuevas['Columna_A'] = pd_distance_altas_nuevas['source_names'].astype(str) + '_' + pd_distance_altas_nuevas['source_municipality'].astype(str) + '_' + pd_distance_altas_nuevas['source_CIF'].astype(str)
    pd_distance_altas_nuevas['Columna_B'] = pd_distance_altas_nuevas['target_names'].astype(str) + '_' + pd_distance_altas_nuevas['target_municipality'].astype(str) + '_' + pd_distance_altas_nuevas['target_CIF'].astype(str)


    Nombres_coinciden = pd_distance_altas_nuevas[['Columna_A', 'Columna_B']]


    source_pd_altas_nuevas = pd_altas_nuevas[['Entidad_Norm', 'CIF', 'Provincia_Entidad']]
    source_pd_altas_nuevas.columns = ['source_names', 'CIF', 'source_municipality']

    target_pd_altas_nuevas = pd_altas_nuevas[['Entidad_Norm', 'CIF', 'Provincia_Entidad']]
    target_pd_altas_nuevas.columns = ['target_names', 'CIF', 'target_municipality']

    NIFs_coinciden = source_pd_altas_nuevas[(target_pd_altas_nuevas['CIF'].notnull()) & (target_pd_altas_nuevas['CIF'] != ' ')].merge(target_pd_altas_nuevas[(target_pd_altas_nuevas['CIF'].notnull()) & (target_pd_altas_nuevas['CIF'] != ' ')],
                                                 how='inner',
                                                  on=['CIF'])
    
    
    print("Registros con CIF = ' '", NIFs_coinciden[NIFs_coinciden['CIF'] == ' '])
    
    NIFs_coinciden = NIFs_coinciden[NIFs_coinciden['CIF'] != ' ']
    
    print("Registros con CIF isnull()", NIFs_coinciden[NIFs_coinciden.CIF.isnull()])
    
    NIFs_coinciden = NIFs_coinciden[NIFs_coinciden.CIF.notnull()]
    
    
    NIFs_coinciden.to_csv(Ruta_Output + 'NIFs_coinciden_Altas_nuevas' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    

    NIFs_coinciden['Columna_A'] = NIFs_coinciden['source_names'].astype(str) + '_' + NIFs_coinciden['source_municipality'].astype(str) + '_' + NIFs_coinciden['CIF'].astype(str)
    NIFs_coinciden['Columna_B'] = NIFs_coinciden['target_names'].astype(str) + '_' + NIFs_coinciden['target_municipality'].astype(str) + '_' + NIFs_coinciden['CIF'].astype(str)

    NIFs_coinciden = NIFs_coinciden[NIFs_coinciden['Columna_A'] != NIFs_coinciden['Columna_B']]

    NIFs_coinciden = NIFs_coinciden[['Columna_A', 'Columna_B']]


    print('Numero de matchs por nombre ', Nombres_coinciden.shape)
    print('Numero de matchs por NIF ', NIFs_coinciden.shape) 

    #CAMBIO IMPORTANTE QUITAMOS LOS MATCHS POR NOMBRE
    #Version con los matchs por nombre
    Nombres_coinciden = pd.concat([Nombres_coinciden, NIFs_coinciden]).values.tolist()
    
    #Version sin los matchs por nombre
    #Nombres_coinciden = pd.concat([NIFs_coinciden]).values.tolist()


    pd_altas_nuevas['Columna'] = pd_altas_nuevas['Entidad_Norm'].astype(str) + '_' + pd_altas_nuevas['Provincia_Entidad'].astype(str) + '_' + pd_altas_nuevas['CIF'].astype(str)

    Nombres_unicos = pd_altas_nuevas[['Columna']].values.tolist()

    resultado_uniones = []
    for i in range(0, len(Nombres_unicos)):
        if(Nombres_unicos[i][0] != ''):
            resultado_uniones.append([Nombres_unicos[i][0]])



    start_time = datetime.datetime.now()


    for i in range(0, len(Nombres_coinciden)):
        Nombres_coinciden[i] = sorted(Remove_duplicate([item for item in Nombres_coinciden[i] if not(pd.isnull(item)) == True]))        


    pd_Nombres_coinciden = pd.DataFrame(Nombres_coinciden)
    
    if(pd_Nombres_coinciden.shape[0] == 0):
        pd_Nombres_coinciden = pd.DataFrame(data = [['Entidad_Inventada_Columna_A', 'Entidad_Inventada_Columna_B']], columns=['Columna_A', 'Columna_B'])
        
    pd_Nombres_coinciden.columns = ['Columna_A', 'Columna_B']

    pd_Nombres_coinciden['cumcount_A'] = pd_Nombres_coinciden.groupby('Columna_A').cumcount() 
    pd_Nombres_coinciden['cumcount_B'] = pd_Nombres_coinciden.groupby('Columna_B').cumcount()

    Vueltas_columna_A = pd_Nombres_coinciden['cumcount_A'].max(axis=0)
    Vueltas_columna_B = pd_Nombres_coinciden['cumcount_B'].max(axis=0)

    print(Vueltas_columna_A, Vueltas_columna_B)
    vueltas = 0
    vuelta_while = 0

    iguales = 1
    fichero_final = []



    resultado_uniones = []
    for i in range(0, len(Nombres_unicos)):
        if(Nombres_unicos[i][0] != ''):
            resultado_uniones.append([Nombres_unicos[i][0]])

    print('Inicio bucle')
    while(iguales == 1):

        rename_columns = []

        pd_resultado_uniones = pd.DataFrame(resultado_uniones)

        for i in pd_resultado_uniones.columns:
            rename_columns.append('Column_' + str(i))
        pd_resultado_uniones.columns = rename_columns

        num_columns = len(pd_resultado_uniones.columns)

        aux = 'Column_' + str(vuelta_while)


        if(aux in pd_resultado_uniones.columns):
            pd_resultado_uniones = pd.DataFrame(resultado_uniones)

            rename_columns = []
            for i in pd_resultado_uniones.columns:
                rename_columns.append('Column_' + str(i))
            pd_resultado_uniones.columns = rename_columns


            resultado_uniones_aux = pd_resultado_uniones[pd_resultado_uniones[aux].isnull()]

            pd_resultado_uniones = pd_resultado_uniones[pd_resultado_uniones[aux].notnull()]


            for j in range(Vueltas_columna_A):
                #print(vueltas)
                vueltas += 1
                pd_resultado_uniones = pd.merge(pd_resultado_uniones, pd_Nombres_coinciden[pd_Nombres_coinciden['cumcount_A'] == j][['Columna_A', 'Columna_B']], how='left', left_on=[aux], right_on=['Columna_A']).drop('Columna_A', axis=1)
                pd_resultado_uniones = pd_resultado_uniones.rename(columns={'Columna_B': 'Columna_B' + str(vueltas)})

            for j in range(Vueltas_columna_B):
                #print(vueltas)
                vueltas +=1
                pd_resultado_uniones = pd.merge(pd_resultado_uniones, pd_Nombres_coinciden[pd_Nombres_coinciden['cumcount_B'] == j][['Columna_A', 'Columna_B']], how='left', left_on=[aux], right_on=['Columna_B']).drop('Columna_B', axis=1)
                pd_resultado_uniones = pd_resultado_uniones.rename(columns={'Columna_A': 'Columna_A' + str(vueltas)})

            resultado_uniones = []
            list_resultado_uniones = pd_resultado_uniones.values.tolist()
            for i in range(len(list_resultado_uniones)):
                resultado_uniones.append(Remove_duplicate([item for item in list_resultado_uniones[i] if not(pd.isnull(item)) == True]))

            list_resultado_uniones_aux = resultado_uniones_aux.values.tolist()
            for i in range(len(list_resultado_uniones_aux)):
                fichero_final.append(sorted(Remove_duplicate([item for item in list_resultado_uniones_aux[i] if not(pd.isnull(item)) == True])))


            vuelta_while += 1
            if(len(resultado_uniones)==0):
                print('len(resultado_uniones)==0')
                iguales = 0

        else:
            print('Fin')
            iguales = 0
            for i in range(len(resultado_uniones)):
                fichero_final.append(sorted(Remove_duplicate([item for item in resultado_uniones[i] if not(pd.isnull(item)) == True])))

        print('Vuelta', vueltas, aux, pd_resultado_uniones.shape, len(resultado_uniones), len(fichero_final))

    print('Tiempo :', datetime.datetime.now() - start_time)

    fichero_final = Remove_duplicate(fichero_final)

    Max_Id = dataframe_output_unificacion4.withColumn('Id', dataframe_output_unificacion4['Id'].cast(IntegerType())).groupby().max('Id').collect()[0][0]
    
    print(Max_Id)
    
    Max_Id = Max_Id + 1
    
    for i in range(0, len(fichero_final)):
        fichero_final[i].insert(0, Max_Id)
        Max_Id += 1

    Tabla_Entidades = pd.DataFrame(fichero_final)

    Tabla_Entidades_id = pd.DataFrame(columns=['Id', 'Entidad_agroup'])

    for i in Tabla_Entidades.columns:
        if(i != 0):
            aux = Tabla_Entidades[[0,i]]
            aux.columns = ['Id', 'Entidad_agroup']
            aux = aux[aux['Entidad_agroup'].notnull()]
            Tabla_Entidades_id = pd.concat([Tabla_Entidades_id, aux])

    Tabla_Entidades_id[['Entidad_Norm','Provincia_Entidad', 'CIF']] = Tabla_Entidades_id.Entidad_agroup.str.split("_",expand=True)

    
    Tabla_Entidades_id = Tabla_Entidades_id[['Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF']]




    print(Tabla_Entidades_id.shape)
    Tabla_Entidades_id = Tabla_Entidades_id.merge(pd_altas_nuevas,
                                                  how = 'left',
                                                 on = ['Entidad_Norm', 'Provincia_Entidad', 'CIF'])


    print(Tabla_Entidades_id.shape)

    Tabla_Entidades_id['Inner'] = 1
    pd_altas_nuevas = pd_altas_nuevas.merge(Tabla_Entidades_id[['Entidad_Norm', 'CIF', 'Provincia_Entidad', 'Inner']],
                                            how='left',
                                            on=['Entidad_Norm', 'CIF', 'Provincia_Entidad'])

    pd_altas_nuevas = pd_altas_nuevas[pd_altas_nuevas['Inner'].isnull()]

    pd_altas_nuevas = pd_altas_nuevas.drop(columns=['Inner'])
    Tabla_Entidades_id = Tabla_Entidades_id.drop(columns=['Inner'])

    
    Tabla_Entidades_id = Tabla_Entidades_id.drop(columns=['Columna'])
    Tabla_Entidades_id['Match'] = 'No Match'



    Tabla_Entidades_id =  Tabla_Entidades_id[['Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Match']]


    Tabla_altas_maestro = Tabla_Entidades_id.drop_duplicates(subset ="Id")[['Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF']]


    Tabla_altas_maestro_aux = Tabla_altas_maestro[['Id', 'Entidad_Norm', 'Provincia_Entidad']]
    Tabla_altas_maestro_aux.columns =['Id', 'Entidad_Match', 'Provincia_Match']

    Tabla_Entidades_id = Tabla_Entidades_id.merge(Tabla_altas_maestro_aux,
                                                  how = 'left',
                                                  on = ['Id'])


    #Hacer el join en pyspark genera error, lo hago desde pandas

    pd_dataframe_output_unificacion5 = dataframe_output_unificacion5.filter(F.col('Id').isNotNull()).toPandas()

    pd_dataframe_output_unificacion5 = pd.concat([pd_dataframe_output_unificacion5, Tabla_altas_maestro])

    print(df_name_dataframe_output_unificacion5)

    #Creamos el esquema para el fichero pd_dataframe_output_unificacion5
    schema = StructType([
        StructField('Id', StringType(), True),
        StructField('CIF', StringType(), True),
        StructField('Entidad_Norm', StringType(), True),
        StructField('Provincia_Entidad', StringType(), True)])

    #Creamos el fichero dataframe_output_unificacion5
    dataframe_output_unificacion5 = spark.createDataFrame(pd_dataframe_output_unificacion5, schema)

    #Casteamos a entero algunos campos
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('Id',dataframe_output_unificacion5.Id.cast(IntegerType()))
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('CIF',dataframe_output_unificacion5.CIF.cast(StringType()))
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('Entidad_Norm',dataframe_output_unificacion5.Entidad_Norm.cast(StringType()))
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('Provincia_Entidad',dataframe_output_unificacion5.Provincia_Entidad.cast(IntegerType()))
    
    
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    
    Dataframe_maestro_orig_aux = Dataframe_maestro_orig.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    Dataframe_invente_orig_aux = Dataframe_invente_orig.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    Dataframe_DIR3_orig_aux = Dataframe_DIR3_orig.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    
    
    #Cambio, repito el código anterior pero incluyendo el ID en el cruce
    #al fichero de unificación le aádo la información de centros y dependencias
    dataframe_output_unificacion5 = dataframe_output_unificacion5.join(Dataframe_maestro_orig_aux.drop('Entidad'),
                                                                       (dataframe_output_unificacion5.Id == Dataframe_maestro_orig_aux.Id) &
                                                             (dataframe_output_unificacion5.Entidad_Norm == Dataframe_maestro_orig_aux.Entidad_Norm) &
                                                             (dataframe_output_unificacion5.Provincia_Entidad == Dataframe_maestro_orig_aux.Provincia_Entidad) &
                                                             (dataframe_output_unificacion5.CIF == Dataframe_maestro_orig_aux.CIF),
                                                             'left').drop(Dataframe_maestro_orig_aux.Entidad_Norm).drop(Dataframe_maestro_orig_aux.Provincia_Entidad).drop(Dataframe_maestro_orig_aux.CIF).drop(Dataframe_maestro_orig_aux.Id)
    

    #al fichero de unificación le aádo la información de Invente
    dataframe_output_unificacion5 = dataframe_output_unificacion5.join(Dataframe_invente_orig_aux.drop('Entidad'),
                                                             (dataframe_output_unificacion5.Entidad_Norm == Dataframe_invente_orig_aux.Entidad_Norm) &
                                                             (dataframe_output_unificacion5.Provincia_Entidad == Dataframe_invente_orig_aux.Provincia_Entidad) &
                                                             (dataframe_output_unificacion5.CIF == Dataframe_invente_orig_aux.CIF),
                                                             'left').drop(Dataframe_invente_orig_aux.Entidad_Norm).drop(Dataframe_invente_orig_aux.Provincia_Entidad).drop(Dataframe_invente_orig_aux.CIF)

    #al fichero de unificación le aádo la información de DIR3
    dataframe_output_unificacion5 = dataframe_output_unificacion5.join(Dataframe_DIR3_orig_aux.drop('Entidad'),
                                                             (dataframe_output_unificacion5.Entidad_Norm == Dataframe_DIR3_orig_aux.Entidad_Norm) &
                                                             (dataframe_output_unificacion5.Provincia_Entidad == Dataframe_DIR3_orig_aux.Provincia_Entidad) &
                                                             (dataframe_output_unificacion5.CIF == Dataframe_DIR3_orig_aux.CIF),
                                                             'left').drop(Dataframe_DIR3_orig_aux.Entidad_Norm).drop(Dataframe_DIR3_orig_aux.Provincia_Entidad).drop(Dataframe_DIR3_orig_aux.CIF)



    #Modificamos el orden de las columnas para que el Id sea el primero
    lista_campos_aux = dataframe_output_unificacion5.columns
    lista_campos_aux.remove('Id')
    lista_campos = ['Id']
    for campo in lista_campos_aux:
        lista_campos.append(campo)
    dataframe_output_unificacion5 = dataframe_output_unificacion5.select(lista_campos)
    
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion5 = dataframe_output_unificacion5.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(' '))

    #Guardamos el fichero
    print('Guardamos dataframe_output_unificacion5')
    dataframe_output_unificacion5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe_output_unificacion5)
    dataframe_output_unificacion5 = spark.read.parquet(Ruta_Output + df_name_dataframe_output_unificacion5)


    
    print(df_name_dataframe_output_unificacion6)

    #Esquema para el fichero Tabla_Entidades_id
    schema = StructType([
        StructField('Id', StringType(), True),
        StructField('Entidad_Norm', StringType(), True),
        StructField('Provincia_Entidad', StringType(), True),
        StructField('CIF', StringType(), True),
        StructField('Match', StringType(), True),
        StructField('Entidad_Match', StringType(), True),
        StructField('Provincia_Match', StringType(), True)])

    #Creamos el fichero Tabla_Entidades_id en dataframe de spark
    pd_Tabla_Entidades_id = spark.createDataFrame(Tabla_Entidades_id, schema).withColumn('Match', F.lit('Alta nueva'))



    print(df_name_dataframe_output_unificacion6)

    #Unimos los ficheros que ya tenemos con Id y los obtenidos en pasos anteriores almacenados en pd_Tabla_Entidades_id
    dataframe_output_unificacion6 = dataframe_output_unificacion6.filter(F.col('Id').isNotNull()).union(pd_Tabla_Entidades_id.select(dataframe_output_unificacion6.columns))

    #Al fichero de unidades organizativas le añadimos la unificación para asociar a cada registro su Id
    dataframe_output_unificacion6 = Dataframe_uo_orig.join(dataframe_output_unificacion6.select('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Id', 'Entidad_Match', 'Provincia_Match', 'Match'),
                                                           (Dataframe_uo_orig.Entidad_Norm == dataframe_output_unificacion6.Entidad_Norm) &
                                                           (Dataframe_uo_orig.Provincia_Entidad == dataframe_output_unificacion6.Provincia_Entidad) &
                                                           (Dataframe_uo_orig.CIF == dataframe_output_unificacion6.CIF),
                                                           'left').drop(dataframe_output_unificacion6.Entidad_Norm).drop(dataframe_output_unificacion6.Provincia_Entidad).drop(dataframe_output_unificacion6.CIF).dropDuplicates()



    #Tenemos un registro que no hace join (aunque todos los valores son iguales), así que este lo tratamos a parte
    dataframe_output_unificacion6_id = dataframe_output_unificacion6.filter(F.col('Id').isNotNull())
    dataframe_output_unificacion6_no_id = dataframe_output_unificacion6.filter(F.col('Id').isNull()).drop('Id')

    #Al registro que no tiene Id repetimos el proceso para intentar asignarle el que le corresponde
    dataframe_output_unificacion6_no_id = dataframe_output_unificacion6_no_id.join(Dataframe_maestro.select('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Id'),
                                                                                   (Dataframe_maestro.Entidad_Norm == dataframe_output_unificacion6_no_id.Entidad_Norm) &
                                                                                   (Dataframe_maestro.Provincia_Entidad == dataframe_output_unificacion6_no_id.Provincia_Entidad) & 
                                                                                   (Dataframe_maestro.CIF == dataframe_output_unificacion6_no_id.CIF),
                                                                                   'left').drop(Dataframe_maestro.Entidad_Norm).drop(Dataframe_maestro.Provincia_Entidad).drop(Dataframe_maestro.CIF).dropDuplicates()

    #Unimos los ficheros con Id Y los que no tenian Id al inicio del proceso (ahora ya debería tener Id asignado)
    dataframe_output_unificacion6 = dataframe_output_unificacion6_id.union(dataframe_output_unificacion6_no_id.select(dataframe_output_unificacion6_id.columns))

    #Como tenemos datos mal leidos, volvemos a repetir el proceso de manera diferente
    dataframe_output_unificacion6_id = dataframe_output_unificacion6.filter(F.col('Id').isNotNull())
    dataframe_output_unificacion6_no_id = dataframe_output_unificacion6.filter(F.col('Id').isNull()).drop('Id')

    #Creamos un fichero auxiliar
    Dataframe_maestro_aux = Dataframe_maestro

    #Casteamos algunos campos en distintos ficheros
    dataframe_output_unificacion6_id = dataframe_output_unificacion6_id.withColumn('Provincia_Entidad',dataframe_output_unificacion6_id.Provincia_Entidad.cast(IntegerType()))
    dataframe_output_unificacion6_no_id = dataframe_output_unificacion6_no_id.withColumn('Provincia_Entidad',dataframe_output_unificacion6_no_id.Provincia_Entidad.cast(IntegerType()))
    Dataframe_maestro_aux = Dataframe_maestro_aux.withColumn('Provincia_Entidad',Dataframe_maestro_aux.Provincia_Entidad.cast(IntegerType()))

    #Intentamos asignarle a los registros sin Id el Id correspondiente
    dataframe_output_unificacion6_no_id = dataframe_output_unificacion6_no_id.join(Dataframe_maestro_aux.select('Entidad_Norm', 'Provincia_Entidad', 'Id'),
                                                                                   (Dataframe_maestro_aux.Entidad_Norm == dataframe_output_unificacion6_no_id.Entidad_Norm) &
                                                                                   (Dataframe_maestro_aux.Provincia_Entidad == dataframe_output_unificacion6_no_id.Provincia_Entidad),
                                                                                   'left').drop(Dataframe_maestro_aux.Entidad_Norm).drop(Dataframe_maestro_aux.Provincia_Entidad).dropDuplicates()

    #Unimos los ficheros con Id Y los que no tenian Id al inicio del proceso (ahora ya debería tener Id asignado)
    dataframe_output_unificacion6 = dataframe_output_unificacion6_id.union(dataframe_output_unificacion6_no_id.select(dataframe_output_unificacion6_id.columns))


    #Modificamos el orden de las columnas para que el Id sea el primero
    lista_campos_aux = dataframe_output_unificacion6.columns
    lista_campos_aux.remove('Id')
    lista_campos = ['Id']
    for campo in lista_campos_aux:
        lista_campos.append(campo)
    dataframe_output_unificacion6 = dataframe_output_unificacion6.select(lista_campos)
    
    
    #Eliminamos duplicados sin tener en cuenta los campos de match
    dataframe_output_unificacion6 = dataframe_output_unificacion6.dropDuplicates(['Id', 'Entidad', 'Origen_Solicitud', 'Identificadores_Origen', 'Entidad_Norm', 'CIF', 'CIF_validacion', 'PIC', 'Tipo', 'Provincia_Entidad', 'Pais_Entidad', 'Centro', 'Centro_Norm', 'Provincia_Centro', 'Match'])

    dataframe_output_unificacion6 = dataframe_output_unificacion6.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion6 = dataframe_output_unificacion6.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion6 = dataframe_output_unificacion6.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion6 = dataframe_output_unificacion6.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion6 = dataframe_output_unificacion6.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    #Guardamos el fichero
    print('Guardamos dataframe_output_unificacion6')
    dataframe_output_unificacion6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe_output_unificacion6)
    dataframe_output_unificacion6 = spark.read.parquet(Ruta_Output + df_name_dataframe_output_unificacion6)

    #Creamos el fichero en pandas
    dataframe_output_unificacion6 = dataframe_output_unificacion6.orderBy(col("Id").desc())
    pd_dataframe_output_unificacion6 = dataframe_output_unificacion6.dropDuplicates().toPandas()

    #casteamos a entero los registros del fichero en pandas
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe_output_unificacion6.columns:
            pd_dataframe_output_unificacion6[column] = np.where(pd_dataframe_output_unificacion6[column]==' ', np.nan, pd_dataframe_output_unificacion6[column])
            pd_dataframe_output_unificacion6[column] = pd_dataframe_output_unificacion6[column].astype('float').astype('Int64')

    #Guardamos el csv obtenido
    pd_dataframe_output_unificacion6.to_csv(Ruta_Output + df_name_dataframe_output_unificacion6 + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    
    #Incluimos los listados en el maestro
    Info_Ids = dataframe_output_unificacion6.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF').union(dataframe_output_unificacion5.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF')).union(Info_Ids_ant.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF')).dropDuplicates()

    Info_Ids = Info_Ids.withColumn('Provincia_Entidad',Info_Ids.Provincia_Entidad.cast(IntegerType()))
    
    Info_Ids.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'Info_Ids')

    Info_Ids = Info_Ids.groupby('Id').agg(F.collect_set("Entidad_Norm").alias('List_Entidad_Norm'),
                                      F.collect_set("Provincia_Entidad").alias('List_Provincia_Entidad'),
                                      F.collect_set("CIF").alias('List_CIF'))

    Info_Ids.dropDuplicates().toPandas().to_csv(Ruta_Output + 'Info_Ids' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    dataframe_output_unificacion5 = dataframe_output_unificacion5.join(Info_Ids,
                                                                  dataframe_output_unificacion5.Id == Info_Ids.Id,
                                                                  'left').drop(Info_Ids.Id)


    
    #Creamos el fichero en Pandas
    dataframe_output_unificacion5 = dataframe_output_unificacion5.orderBy(col("Id").desc())
    
    pd_dataframe_output_unificacion5 = dataframe_output_unificacion5.dropDuplicates().toPandas()

    #Casteamos a entero algunas de las columnas
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe_output_unificacion5.columns:
            pd_dataframe_output_unificacion5[column] = np.where(pd_dataframe_output_unificacion5[column]==' ', np.nan, pd_dataframe_output_unificacion5[column])
            pd_dataframe_output_unificacion5[column] = pd_dataframe_output_unificacion5[column].astype('float').astype('Int64')

    #Guardamos el fichero en csv
    pd_dataframe_output_unificacion5.to_csv(Ruta_Output + df_name_dataframe_output_unificacion5 + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)

    
    
    

    #Comprobacion para ver si tenemos registros sin Id (primera parte) o registros con distintos Id (segunda parte)

    print(dataframe_output_unificacion6.filter(F.col('Id').isNull()).groupBy("Entidad_Norm").count().count())
    dataframe_output_unificacion6.filter(F.col('Id').isNull()).groupBy("Entidad_Norm").count().show(100, False)

    print(dataframe_output_unificacion6.groupBy('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Id').count().groupBy('Entidad_Norm', 'Provincia_Entidad', 'CIF').count().filter(F.col('count') > 1).count())
    dataframe_output_unificacion6.groupBy('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Id').count().groupBy('Entidad_Norm', 'Provincia_Entidad', 'CIF').count().filter(F.col('count') > 1).show(100, False)





    #Guardamos copia de seguridad de todos los ficheros

    print('dataframe1_output_cruzan_cruce1')
    dataframe1_output_cruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1_cs')

    print('dataframe2_output_nocruzan_cruce1')
    dataframe2_output_nocruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1_cs')

    print('dataframe1_output_cruzan_cruce2')
    dataframe1_output_cruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2_cs')

    print('dataframe2_output_nocruzan_cruce2')
    dataframe2_output_nocruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2_cs')

    print('dataframe1_output_cruzan_cruce3')
    dataframe1_output_cruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3_cs')

    print('dataframe2_output_nocruzan_cruce3')
    dataframe2_output_nocruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3_cs')

    print('dataframe1_output_cruzan_cruce4')
    dataframe1_output_cruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4_cs')

    print('dataframe2_output_nocruzan_cruce4')
    dataframe2_output_nocruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4_cs')

    print('dataframe1_output_cruzan_cruce5')
    dataframe1_output_cruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5_cs')

    print('dataframe2_output_nocruzan_cruce5')
    dataframe2_output_nocruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5_cs')

    print('dataframe1_output_cruzan_cruce6')
    dataframe1_output_cruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6_cs')

    print('dataframe2_output_nocruzan_cruce6')
    dataframe2_output_nocruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6_cs')



    print('dataframe_output_unificacion1')
    dataframe_output_unificacion1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion1_cs')

    print('dataframe_output_unificacion2')
    dataframe_output_unificacion2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion2_cs')

    print('dataframe_output_unificacion3')
    dataframe_output_unificacion3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion3_cs')

    print('dataframe_output_unificacion4')
    dataframe_output_unificacion4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion4_cs')

    print('dataframe_output_unificacion5')
    dataframe_output_unificacion5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion5_cs')

    print('dataframe_output_unificacion6')
    dataframe_output_unificacion6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion6_cs')





    #Leemos copia de seguridad de todos los ficheros

    print('dataframe1_output_cruzan_cruce1')
    dataframe1_output_cruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1_cs')

    print('dataframe2_output_nocruzan_cruce1')
    dataframe2_output_nocruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1_cs')

    print('dataframe1_output_cruzan_cruce2')
    dataframe1_output_cruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2_cs')

    print('dataframe2_output_nocruzan_cruce2')
    dataframe2_output_nocruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2_cs')

    print('dataframe1_output_cruzan_cruce3')
    dataframe1_output_cruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3_cs')

    print('dataframe2_output_nocruzan_cruce3')
    dataframe2_output_nocruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3_cs')

    print('dataframe1_output_cruzan_cruce4')
    dataframe1_output_cruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4_cs')

    print('dataframe2_output_nocruzan_cruce4')
    dataframe2_output_nocruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4_cs')

    print('dataframe1_output_cruzan_cruce5')
    dataframe1_output_cruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5_cs')

    print('dataframe2_output_nocruzan_cruce5')
    dataframe2_output_nocruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5_cs')

    print('dataframe1_output_cruzan_cruce6')
    dataframe1_output_cruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6_cs')

    print('dataframe2_output_nocruzan_cruce6')
    dataframe2_output_nocruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6_cs')



    print('dataframe_output_unificacion1')
    dataframe_output_unificacion1 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion1_cs')

    print('dataframe_output_unificacion2')
    dataframe_output_unificacion2 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion2_cs')

    print('dataframe_output_unificacion3')
    dataframe_output_unificacion3 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion3_cs')

    print('dataframe_output_unificacion4')
    dataframe_output_unificacion4 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion4_cs')

    print('dataframe_output_unificacion5')
    dataframe_output_unificacion5 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion5_cs')

    print('dataframe_output_unificacion6')
    dataframe_output_unificacion6 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion6_cs')



    #Creamos un fichero con los Id para poder añadirlo a los cruces
    ids = dataframe_output_unificacion6.select('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Id').dropDuplicates()


    #Creamos un fichero con los Id para poder añadirlo a los cruces
    ids_maestro = dataframe_output_unificacion5.select('Entidad_Norm', 'Provincia_Entidad', 'CIF', 'Id').dropDuplicates()
    #ids_maestro = ids_maestro.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(9999))
    
    
    ids_maestro = ids_maestro.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad').isNotNull(), F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    ids_maestro = ids_maestro.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(F.lit(' ')))
    ids_maestro = ids_maestro.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'nan', F.col('Provincia_Entidad')).otherwise(F.lit(' ')))

    ids_maestro = ids_maestro.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise(F.lit(' ')))
    ids_maestro = ids_maestro.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(' ')))
    ids_maestro = ids_maestro.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(' ')))
    
    
    
    ids_maestro= ids_maestro.withColumn("Provincia_Entidad",col("Provincia_Entidad").cast("integer"))

    
    ids.dropDuplicates().toPandas().to_csv(Ruta_Output + 'ids' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    ids_maestro.dropDuplicates().toPandas().to_csv(Ruta_Output + 'ids_maestro' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    
    
    dataframe1_output_cruzan_cruce1_export = dataframe_output_unificacion6.filter(F.col('Match') == 'NIF MCIN')

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe1_output_cruzan_cruce1_export,
                 df_name_dataframe1_output_cruzan_cruce1)


    dataframe2_output_nocruzan_cruce1_export = dataframe_output_unificacion6.filter(F.col('Match') != 'NIF MCIN').drop('Entidad_Match').drop('Provincia_Match').drop('Match').drop('Id').dropDuplicates()

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe2_output_nocruzan_cruce1_export,
                 df_name_dataframe2_output_nocruzan_cruce1)


    dataframe1_output_cruzan_cruce2_export = dataframe_output_unificacion6.filter(F.col('Match') == 'NIF INVENTE')

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe1_output_cruzan_cruce2_export,
                 df_name_dataframe1_output_cruzan_cruce2)


    dataframe2_output_nocruzan_cruce2_export = dataframe_output_unificacion6.filter((F.col('Match') != 'NIF MCIN') &
                                                                                    (F.col('Match') != 'NIF INVENTE')).drop('Entidad_Match').drop('Provincia_Match').drop('Match').drop('Id').dropDuplicates()

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe2_output_nocruzan_cruce2_export,
                 df_name_dataframe2_output_nocruzan_cruce2)


    dataframe1_output_cruzan_cruce3_export = dataframe_output_unificacion6.filter(F.col('Match') == 'NIF DIR3')

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe1_output_cruzan_cruce3_export,
                 df_name_dataframe1_output_cruzan_cruce3)


    dataframe2_output_nocruzan_cruce3_export = dataframe_output_unificacion6.filter((F.col('Match') != 'NIF MCIN') &
                                                                                    (F.col('Match') != 'NIF INVENTE') &
                                                                                    (F.col('Match') != 'NIF DIR3')).drop('Entidad_Match').drop('Provincia_Match').drop('Match').drop('Id').dropDuplicates()

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe2_output_nocruzan_cruce3_export,
                 df_name_dataframe2_output_nocruzan_cruce3)


    dataframe1_output_cruzan_cruce4_export = dataframe_output_unificacion6.filter(F.col('Match') == 'NOMBRE MCIN')

    dataframe1_output_cruzan_cruce4_export = dataframe1_output_cruzan_cruce4_export.join(dataframe1_output_cruzan_cruce4.select('Entidad_Norm', 'Entidad_Match', 'final_score'),
                                               (dataframe1_output_cruzan_cruce4_export.Entidad_Norm == dataframe1_output_cruzan_cruce4.Entidad_Norm) &
                                               (dataframe1_output_cruzan_cruce4_export.Entidad_Match == dataframe1_output_cruzan_cruce4.Entidad_Match)).drop(dataframe1_output_cruzan_cruce4.Entidad_Norm).drop(dataframe1_output_cruzan_cruce4.Entidad_Match)

    dataframe1_output_cruzan_cruce4_export = dataframe1_output_cruzan_cruce4_export.select('Id', 'Entidad', 'Origen_Solicitud', 'Identificadores_Origen', 'Entidad_Norm', 'CIF', 'CIF_validacion', 'PIC', 'Tipo', 'Provincia_Entidad', 'Pais_Entidad', 'Centro', 'Centro_Norm', 'Provincia_Centro', 'Tipo_Persona', 'Entidad_Match', 'Provincia_Match', 'final_score', 'Match')

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe1_output_cruzan_cruce4_export,
                 df_name_dataframe1_output_cruzan_cruce4)


    dataframe2_output_nocruzan_cruce4_export = dataframe_output_unificacion6.filter((F.col('Match') != 'NIF MCIN') &
                                                                                    (F.col('Match') != 'NIF INVENTE') &
                                                                                    (F.col('Match') != 'NIF DIR3') &
                                                                                    (F.col('Match') != 'NOMBRE MCIN')).drop('Entidad_Match').drop('Provincia_Match').drop('Match').drop('Id').dropDuplicates()

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe2_output_nocruzan_cruce4_export,
                 df_name_dataframe2_output_nocruzan_cruce4)



    dataframe1_output_cruzan_cruce5_export = dataframe_output_unificacion6.filter(F.col('Match') == 'NOMBRE INVENTE')

    dataframe1_output_cruzan_cruce5_export = dataframe1_output_cruzan_cruce5_export.join(dataframe1_output_cruzan_cruce5.select('Entidad_Norm', 'Entidad_Match', 'final_score'),
                                               (dataframe1_output_cruzan_cruce5_export.Entidad_Norm == dataframe1_output_cruzan_cruce5.Entidad_Norm) &
                                               (dataframe1_output_cruzan_cruce5_export.Entidad_Match == dataframe1_output_cruzan_cruce5.Entidad_Match)).drop(dataframe1_output_cruzan_cruce5.Entidad_Norm).drop(dataframe1_output_cruzan_cruce5.Entidad_Match)

    dataframe1_output_cruzan_cruce5_export = dataframe1_output_cruzan_cruce5_export.select('Id', 'Entidad', 'Origen_Solicitud', 'Identificadores_Origen', 'Entidad_Norm', 'CIF', 'CIF_validacion', 'PIC', 'Tipo', 'Provincia_Entidad', 'Pais_Entidad', 'Centro', 'Centro_Norm', 'Provincia_Centro', 'Tipo_Persona', 'Entidad_Match', 'Provincia_Match', 'final_score', 'Match')

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe1_output_cruzan_cruce5_export,
                 df_name_dataframe1_output_cruzan_cruce5)



    dataframe2_output_nocruzan_cruce5_export = dataframe_output_unificacion6.filter((F.col('Match') != 'NIF MCIN') &
                                                                                    (F.col('Match') != 'NIF INVENTE') &
                                                                                    (F.col('Match') != 'NIF DIR3') &
                                                                                    (F.col('Match') != 'NOMBRE MCIN') &
                                                                                    (F.col('Match') != 'NOMBRE INVENTE')).drop('Entidad_Match').drop('Provincia_Match').drop('Match').drop('Id').dropDuplicates()

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe2_output_nocruzan_cruce5_export,
                 df_name_dataframe2_output_nocruzan_cruce5)



    dataframe1_output_cruzan_cruce6_export = dataframe_output_unificacion6.filter(F.col('Match') == 'NOMBRE DIR3')

    dataframe1_output_cruzan_cruce6_export = dataframe1_output_cruzan_cruce6_export.join(dataframe1_output_cruzan_cruce6.select('Entidad_Norm', 'Entidad_Match', 'final_score'),
                                               (dataframe1_output_cruzan_cruce6_export.Entidad_Norm == dataframe1_output_cruzan_cruce6.Entidad_Norm) &
                                               (dataframe1_output_cruzan_cruce6_export.Entidad_Match == dataframe1_output_cruzan_cruce6.Entidad_Match)).drop(dataframe1_output_cruzan_cruce6.Entidad_Norm).drop(dataframe1_output_cruzan_cruce6.Entidad_Match)

    dataframe1_output_cruzan_cruce6_export = dataframe1_output_cruzan_cruce6_export.select('Id', 'Entidad', 'Origen_Solicitud', 'Identificadores_Origen', 'Entidad_Norm', 'CIF', 'CIF_validacion', 'PIC', 'Tipo', 'Provincia_Entidad', 'Pais_Entidad', 'Centro', 'Centro_Norm', 'Provincia_Centro', 'Tipo_Persona', 'Entidad_Match', 'Provincia_Match', 'final_score', 'Match')

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe1_output_cruzan_cruce6_export,
                 df_name_dataframe1_output_cruzan_cruce6)




    dataframe2_output_nocruzan_cruce6_export = dataframe_output_unificacion6.filter((F.col('Match') != 'NIF MCIN') &
                                                                                    (F.col('Match') != 'NIF INVENTE') &
                                                                                    (F.col('Match') != 'NIF DIR3') &
                                                                                    (F.col('Match') != 'NOMBRE MCIN') &
                                                                                    (F.col('Match') != 'NOMBRE INVENTE') &
                                                                                    (F.col('Match') != 'NOMBRE DIR3')).drop('Entidad_Match').drop('Provincia_Match').drop('Match').drop('Id').dropDuplicates()

    exportar_csv(spark,
                 Ruta_Output,
                 dataframe2_output_nocruzan_cruce6_export,
                 df_name_dataframe2_output_nocruzan_cruce6)





    #Creamos un fichero modificando los CIFS nulos por 'NONE' para intentar solucionar datos con errores
    Dataframe_maestro_orig_None = Dataframe_maestro_orig
    Dataframe_maestro_orig_None = Dataframe_maestro_orig_None.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))

    #Creamos un fichero modificando los CIFS nulos por 'NONE' para intentar solucionar datos con errores
    Dataframe_invente_orig_None = Dataframe_invente_orig
    Dataframe_invente_orig_None = Dataframe_invente_orig_None.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))

    #Creamos un fichero modificando los CIFS nulos por 'NONE' para intentar solucionar datos con errores
    Dataframe_DIR3_orig_None = Dataframe_DIR3_orig
    Dataframe_DIR3_orig_None = Dataframe_DIR3_orig_None.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))

    #Creamos ficheros modificando los CIFS nulos por 'NONE' para intentar solucionar datos con errores
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('CIF', F.when(F.col('CIF').isNotNull(), F.col('CIF')).otherwise('None'))







    print(df_name_dataframe_output_unificacion1)

    #Añadimos la informacion del fichero de centros y dependencias
    dataframe_output_unificacion1 = dataframe_output_unificacion1.join(Dataframe_maestro_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion1.Id == Dataframe_maestro_orig_None.Id) &
                                                                       (dataframe_output_unificacion1.Entidad_Norm == Dataframe_maestro_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion1.Provincia_Entidad == Dataframe_maestro_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion1.CIF == Dataframe_maestro_orig_None.CIF),
                                                                       'left').drop(Dataframe_maestro_orig_None.Id).drop(Dataframe_maestro_orig_None.Entidad_Norm).drop(Dataframe_maestro_orig_None.Provincia_Entidad).drop(Dataframe_maestro_orig_None.CIF)

    #Añadimos la informacion del fichero de Invente
    dataframe_output_unificacion1 = dataframe_output_unificacion1.join(Dataframe_invente_orig_None.drop('Entidad'),
                                                             (dataframe_output_unificacion1.Entidad_Norm == Dataframe_invente_orig_None.Entidad_Norm) &
                                                             (dataframe_output_unificacion1.Provincia_Entidad == Dataframe_invente_orig_None.Provincia_Entidad) &
                                                             (dataframe_output_unificacion1.CIF == Dataframe_invente_orig_None.CIF),
                                                             'left').drop(Dataframe_invente_orig_None.Entidad_Norm).drop(Dataframe_invente_orig_None.Provincia_Entidad).drop(Dataframe_invente_orig_None.CIF)

    #Añadimos la columna Entidad
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn("Entidad", F.when(F.col('Nombre_Entidad_Mostrar').isNotNull(),F.col('Nombre_Entidad_Mostrar'))                                                                          .otherwise(F.col('DenominacionSocial')))

    #Tenemos registros con errores, esto lo hacemos para arreglar el problema (aunque no siempre funciona, en casos puntuales hemos comprobado que si)
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('CIF', F.when(F.col('CIF') != 'None', F.col('CIF')).otherwise(None))

    #Modificamos el orden de las columnas para que el Id sea el primero
    dataframe_output_unificacion1 = dataframe_output_unificacion1.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'ID_ENTIDAD', 'NIF_COD', 'ACRONIMO', 'NOMBRE_ENTIDAD', 'Nombre_Entidad_Mostrar', 'TIPO_ENTIDAD_N1_1', 'TIPO_ENTIDAD_N2_1', 'DIRECCION_POSTAL', 'COD_POSTAL', 'COD_PROVINCIA', 'PROVINCIA', 'COD_CCAA', 'CCAA', 'ENLACE_WEB', 'SOMMA', 'TIPO_ENTIDAD_REGIONAL', 'ESTADO_x', 'CodigoInvente', 'DenominacionSocial', 'FormaJuridica_Codigo', 'FormaJuridica_Descripcion', 'NIF', 'codigoDir3', 'codigoOrigen', 'Provincia_Codigo')

    #Casteamos las columnas correspondientes a valor entero
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('Provincia_Entidad',dataframe_output_unificacion1.Provincia_Entidad.cast(IntegerType()))
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('ID_ENTIDAD',dataframe_output_unificacion1.ID_ENTIDAD.cast(IntegerType()))
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('COD_POSTAL',dataframe_output_unificacion1.COD_POSTAL.cast(IntegerType()))
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('COD_PROVINCIA',dataframe_output_unificacion1.COD_PROVINCIA.cast(IntegerType()))
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('COD_CCAA',dataframe_output_unificacion1.COD_CCAA.cast(IntegerType()))
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('Provincia_Codigo',dataframe_output_unificacion1.Provincia_Codigo.cast(IntegerType()))

    #Guardamos los parquets
    dataframe_output_unificacion1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe_output_unificacion1)
    dataframe_output_unificacion1 = spark.read.parquet(Ruta_Output + df_name_dataframe_output_unificacion1)
    
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))

    
    dataframe_output_unificacion1 = dataframe_output_unificacion1.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    
    #Creamos el fichero pandas
    pd_dataframe_output_unificacion1 = dataframe_output_unificacion1.dropDuplicates().toPandas()

    #Casteamos a enteros los valores necesarios
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe_output_unificacion1.columns:
            pd_dataframe_output_unificacion1[column] = np.where(pd_dataframe_output_unificacion1[column]==' ', np.nan, pd_dataframe_output_unificacion1[column])
            pd_dataframe_output_unificacion1[column] = pd_dataframe_output_unificacion1[column].astype('float').astype('Int64')

    #Eliminamos los saltos de carro
    pd_dataframe_output_unificacion1 = pd_dataframe_output_unificacion1.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_dataframe_output_unificacion1.to_csv(Ruta_Output + df_name_dataframe_output_unificacion1 + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(dataframe_output_unificacion1.count())
    dataframe_output_unificacion1.show()











    print(df_name_dataframe_output_unificacion2)

    #Añadimos la informacion del fichero de centros y dependencias
    dataframe_output_unificacion2 = dataframe_output_unificacion2.join(Dataframe_maestro_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion2.Id == Dataframe_maestro_orig_None.Id) &
                                                                       (dataframe_output_unificacion2.Entidad_Norm == Dataframe_maestro_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion2.Provincia_Entidad == Dataframe_maestro_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion2.CIF == Dataframe_maestro_orig_None.CIF),
                                                                       'left').drop(Dataframe_maestro_orig_None.Id).drop(Dataframe_maestro_orig_None.Entidad_Norm).drop(Dataframe_maestro_orig_None.Provincia_Entidad).drop(Dataframe_maestro_orig_None.CIF)

    #Añadimos la informacion del fichero de Invente
    dataframe_output_unificacion2 = dataframe_output_unificacion2.join(Dataframe_invente_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion2.Entidad_Norm == Dataframe_invente_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion2.Provincia_Entidad == Dataframe_invente_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion2.CIF == Dataframe_invente_orig_None.CIF),
                                                                       'left').drop(Dataframe_invente_orig_None.Entidad_Norm).drop(Dataframe_invente_orig_None.Provincia_Entidad).drop(Dataframe_invente_orig_None.CIF)

    #Añadimos la informacion del fichero de DIR3
    dataframe_output_unificacion2 = dataframe_output_unificacion2.join(Dataframe_DIR3_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion2.Entidad_Norm == Dataframe_DIR3_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion2.Provincia_Entidad == Dataframe_DIR3_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion2.CIF == Dataframe_DIR3_orig_None.CIF),
                                                                       'left').drop(Dataframe_DIR3_orig_None.Entidad_Norm).drop(Dataframe_DIR3_orig_None.Provincia_Entidad).drop(Dataframe_DIR3_orig_None.CIF)

    #Añadimos la columna Entidad
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn("Entidad", F.when(F.col('Nombre_Entidad_Mostrar').isNotNull(),F.col('Nombre_Entidad_Mostrar'))                                                                          .when(F.col('DenominacionSocial').isNotNull(), F.col('DenominacionSocial'))
                                                                             .otherwise(F.col('C_DNM_UD_ORGANICA')))

    #Tenemos registros con errores, esto lo hacemos para arreglar el problema (aunque no siempre funciona, en casos puntuales hemos comprobado que si)
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('CIF', F.when(F.col('CIF') != 'None', F.col('CIF')).otherwise(None))

    #Modificamos el orden de las columnas para que el Id sea el primero
    dataframe_output_unificacion2 = dataframe_output_unificacion2.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'ID_ENTIDAD', 'NIF_COD', 'ACRONIMO', 'NOMBRE_ENTIDAD', 'Nombre_Entidad_Mostrar', 'TIPO_ENTIDAD_N1_1', 'TIPO_ENTIDAD_N2_1', 'DIRECCION_POSTAL', 'COD_POSTAL', 'COD_PROVINCIA', 'PROVINCIA', 'COD_CCAA', 'CCAA', 'ENLACE_WEB', 'SOMMA', 'TIPO_ENTIDAD_REGIONAL', 'ESTADO_x', 'CodigoInvente', 'DenominacionSocial', 'FormaJuridica_Codigo', 'FormaJuridica_Descripcion', 'NIF', 'codigoDir3', 'codigoOrigen', 'Provincia_Codigo', 'C_ID_UD_ORGANICA', 'C_DNM_UD_ORGANICA', 'C_ID_NIVEL_ADMON', 'C_ID_TIPO_ENT_PUBLICA', 'N_NIVEL_JERARQUICO', 'C_ID_DEP_UD_SUPERIOR', 'C_DNM_UD_ORGANICA_SUPERIOR', 'C_ID_DEP_UD_PRINCIPAL', 'C_DNM_UD_ORGANICA_PRINCIPAL', 'B_SW_DEP_EDP_PRINCIPAL', 'C_ID_DEP_EDP_PRINCIPAL', 'C_DNM_UD_ORGANICA_EDP_PRINCIPAL', 'C_ID_ESTADO', 'D_VIG_ALTA_OFICIAL', 'NIF_CIF', 'C_ID_AMB_PROVINCIA', 'C_DESC_PROV', 'CONTACTOS')

    #Casteamos las columnas correspondientes a valor entero
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('Provincia_Entidad',dataframe_output_unificacion2.Provincia_Entidad.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('ID_ENTIDAD',dataframe_output_unificacion2.ID_ENTIDAD.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('COD_POSTAL',dataframe_output_unificacion2.COD_POSTAL.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('COD_PROVINCIA',dataframe_output_unificacion2.COD_PROVINCIA.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('COD_CCAA',dataframe_output_unificacion2.COD_CCAA.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('Provincia_Codigo',dataframe_output_unificacion2.Provincia_Codigo.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('C_ID_NIVEL_ADMON',dataframe_output_unificacion2.C_ID_NIVEL_ADMON.cast(IntegerType()))
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('N_NIVEL_JERARQUICO',dataframe_output_unificacion2.N_NIVEL_JERARQUICO.cast(IntegerType()))

    #Guardamos los parquets
    dataframe_output_unificacion2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe_output_unificacion2)
    dataframe_output_unificacion2 = spark.read.parquet(Ruta_Output + df_name_dataframe_output_unificacion2)
    
    
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion2 = dataframe_output_unificacion2.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))

    #Creamos el fichero pandas
    pd_dataframe_output_unificacion2 = dataframe_output_unificacion2.dropDuplicates().toPandas()

    #Casteamos a enteros los valores necesarios
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe_output_unificacion2.columns:
            pd_dataframe_output_unificacion2[column] = np.where(pd_dataframe_output_unificacion2[column]==' ', np.nan, pd_dataframe_output_unificacion2[column])
            pd_dataframe_output_unificacion2[column] = pd_dataframe_output_unificacion2[column].astype('float').astype('Int64')

    #Eliminamos los saltos de carro
    pd_dataframe_output_unificacion2 = pd_dataframe_output_unificacion2.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_dataframe_output_unificacion2.to_csv(Ruta_Output + df_name_dataframe_output_unificacion2 + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(dataframe_output_unificacion2.count())
    dataframe_output_unificacion2.show()











    print(df_name_dataframe_output_unificacion3)

    #Añadimos la informacion del fichero de centros y dependencias
    dataframe_output_unificacion3 = dataframe_output_unificacion3.join(Dataframe_maestro_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion3.Id == Dataframe_maestro_orig_None.Id) &
                                                                       (dataframe_output_unificacion3.Entidad_Norm == Dataframe_maestro_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion3.Provincia_Entidad == Dataframe_maestro_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion3.CIF == Dataframe_maestro_orig_None.CIF),
                                                                       'left').drop(Dataframe_maestro_orig_None.Id).drop(Dataframe_maestro_orig_None.Entidad_Norm).drop(Dataframe_maestro_orig_None.Provincia_Entidad).drop(Dataframe_maestro_orig_None.CIF)

    #Añadimos la informacion del fichero de Invente
    dataframe_output_unificacion3 = dataframe_output_unificacion3.join(Dataframe_invente_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion3.Entidad_Norm == Dataframe_invente_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion3.Provincia_Entidad == Dataframe_invente_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion3.CIF == Dataframe_invente_orig_None.CIF),
                                                                       'left').drop(Dataframe_invente_orig_None.Entidad_Norm).drop(Dataframe_invente_orig_None.Provincia_Entidad).drop(Dataframe_invente_orig_None.CIF)

    #Añadimos la informacion del fichero de DIR3
    dataframe_output_unificacion3 = dataframe_output_unificacion3.join(Dataframe_DIR3_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion3.Entidad_Norm == Dataframe_DIR3_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion3.Provincia_Entidad == Dataframe_DIR3_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion3.CIF == Dataframe_DIR3_orig_None.CIF),
                                                                       'left').drop(Dataframe_DIR3_orig_None.Entidad_Norm).drop(Dataframe_DIR3_orig_None.Provincia_Entidad).drop(Dataframe_DIR3_orig_None.CIF)

    #Añadimos la columna Entidad
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn("Entidad", F.when(F.col('Nombre_Entidad_Mostrar').isNotNull(),F.col('Nombre_Entidad_Mostrar'))                                                                          .when(F.col('DenominacionSocial').isNotNull(), F.col('DenominacionSocial'))
                                                                             .otherwise(F.col('C_DNM_UD_ORGANICA')))

    #Tenemos registros con errores, esto lo hacemos para arreglar el problema (aunque no siempre funciona, en casos puntuales hemos comprobado que si)
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('CIF', F.when(F.col('CIF') != 'None', F.col('CIF')).otherwise(None))

    #Seleccionamos las columnas en el orden deseado
    dataframe_output_unificacion3 = dataframe_output_unificacion3.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'ID_ENTIDAD', 'NIF_COD', 'ACRONIMO', 'NOMBRE_ENTIDAD', 'Nombre_Entidad_Mostrar', 'TIPO_ENTIDAD_N1_1', 'TIPO_ENTIDAD_N2_1', 'DIRECCION_POSTAL', 'COD_POSTAL', 'COD_PROVINCIA', 'PROVINCIA', 'COD_CCAA', 'CCAA', 'ENLACE_WEB', 'SOMMA', 'TIPO_ENTIDAD_REGIONAL', 'ESTADO_x', 'CodigoInvente', 'DenominacionSocial', 'FormaJuridica_Codigo', 'FormaJuridica_Descripcion', 'NIF', 'codigoDir3', 'codigoOrigen', 'Provincia_Codigo', 'C_ID_UD_ORGANICA', 'C_DNM_UD_ORGANICA', 'C_ID_NIVEL_ADMON', 'C_ID_TIPO_ENT_PUBLICA', 'N_NIVEL_JERARQUICO', 'C_ID_DEP_UD_SUPERIOR', 'C_DNM_UD_ORGANICA_SUPERIOR', 'C_ID_DEP_UD_PRINCIPAL', 'C_DNM_UD_ORGANICA_PRINCIPAL','B_SW_DEP_EDP_PRINCIPAL', 'C_ID_DEP_EDP_PRINCIPAL', 'C_DNM_UD_ORGANICA_EDP_PRINCIPAL', 'C_ID_ESTADO', 'D_VIG_ALTA_OFICIAL', 'NIF_CIF', 'C_ID_AMB_PROVINCIA', 'C_DESC_PROV', 'CONTACTOS')

    #Casteamos las columnas correspondientes a valor entero
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('Provincia_Entidad',dataframe_output_unificacion3.Provincia_Entidad.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('ID_ENTIDAD',dataframe_output_unificacion3.ID_ENTIDAD.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('COD_POSTAL',dataframe_output_unificacion3.COD_POSTAL.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('COD_PROVINCIA',dataframe_output_unificacion3.COD_PROVINCIA.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('COD_CCAA',dataframe_output_unificacion3.COD_CCAA.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('Provincia_Codigo',dataframe_output_unificacion3.Provincia_Codigo.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('C_ID_NIVEL_ADMON',dataframe_output_unificacion3.C_ID_NIVEL_ADMON.cast(IntegerType()))
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('N_NIVEL_JERARQUICO',dataframe_output_unificacion3.N_NIVEL_JERARQUICO.cast(IntegerType()))

    #Guardamos los parquets
    dataframe_output_unificacion3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe_output_unificacion3)
    dataframe_output_unificacion3 = spark.read.parquet(Ruta_Output + df_name_dataframe_output_unificacion3)
    
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    
    dataframe_output_unificacion3 = dataframe_output_unificacion3.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))

    #Creamos el fichero pandas
    pd_dataframe_output_unificacion3 = dataframe_output_unificacion3.dropDuplicates().toPandas()

    #Casteamos a enteros los valores necesarios
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe_output_unificacion3.columns:
            pd_dataframe_output_unificacion3[column] = np.where(pd_dataframe_output_unificacion3[column]==' ', np.nan, pd_dataframe_output_unificacion3[column])
            pd_dataframe_output_unificacion3[column] = pd_dataframe_output_unificacion3[column].astype('float').astype('Int64')

    #Eliminamos los saltos de carro
    pd_dataframe_output_unificacion3 = pd_dataframe_output_unificacion3.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_dataframe_output_unificacion3.to_csv(Ruta_Output + df_name_dataframe_output_unificacion3 + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(dataframe_output_unificacion3.count())
    dataframe_output_unificacion3.show()











    print(df_name_dataframe_output_unificacion4)

    #Añadimos la informacion del fichero de centros y dependencias
    dataframe_output_unificacion4 = dataframe_output_unificacion4.join(Dataframe_maestro_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion4.Id == Dataframe_maestro_orig_None.Id) &
                                                                       (dataframe_output_unificacion4.Entidad_Norm == Dataframe_maestro_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion4.Provincia_Entidad == Dataframe_maestro_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion4.CIF == Dataframe_maestro_orig_None.CIF),
                                                                       'left').drop(Dataframe_maestro_orig_None.Id).drop(Dataframe_maestro_orig_None.Entidad_Norm).drop(Dataframe_maestro_orig_None.Provincia_Entidad).drop(Dataframe_maestro_orig_None.CIF)
    #Añadimos la informacion del fichero de Invente
    dataframe_output_unificacion4 = dataframe_output_unificacion4.join(Dataframe_invente_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion4.Entidad_Norm == Dataframe_invente_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion4.Provincia_Entidad == Dataframe_invente_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion4.CIF == Dataframe_invente_orig_None.CIF),
                                                                       'left').drop(Dataframe_invente_orig_None.Entidad_Norm).drop(Dataframe_invente_orig_None.Provincia_Entidad).drop(Dataframe_invente_orig_None.CIF)

    #Añadimos la informacion del fichero de DIR3
    dataframe_output_unificacion4 = dataframe_output_unificacion4.join(Dataframe_DIR3_orig_None.drop('Entidad'),
                                                                       (dataframe_output_unificacion4.Entidad_Norm == Dataframe_DIR3_orig_None.Entidad_Norm) &
                                                                       (dataframe_output_unificacion4.Provincia_Entidad == Dataframe_DIR3_orig_None.Provincia_Entidad) &
                                                                       (dataframe_output_unificacion4.CIF == Dataframe_DIR3_orig_None.CIF),
                                                                       'left').drop(Dataframe_DIR3_orig_None.Entidad_Norm).drop(Dataframe_DIR3_orig_None.Provincia_Entidad).drop(Dataframe_DIR3_orig_None.CIF)

    #Añadimos la columna Entidad
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn("Entidad", F.when(F.col('Nombre_Entidad_Mostrar').isNotNull(),F.col('Nombre_Entidad_Mostrar'))                                                                          .when(F.col('DenominacionSocial').isNotNull(), F.col('DenominacionSocial'))
                                                                             .otherwise(F.col('C_DNM_UD_ORGANICA')))

    #Tenemos registros con errores, esto lo hacemos para arreglar el problema (aunque no siempre funciona, en casos puntuales hemos comprobado que si)
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('CIF', F.when(F.col('CIF') != 'None', F.col('CIF')).otherwise(None))

    #Seleccionamos las columnas en el orden deseado
    dataframe_output_unificacion4 = dataframe_output_unificacion4.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'ID_ENTIDAD', 'NIF_COD', 'ACRONIMO', 'NOMBRE_ENTIDAD', 'Nombre_Entidad_Mostrar', 'TIPO_ENTIDAD_N1_1', 'TIPO_ENTIDAD_N2_1', 'DIRECCION_POSTAL', 'COD_POSTAL', 'COD_PROVINCIA', 'PROVINCIA', 'COD_CCAA', 'CCAA', 'ENLACE_WEB', 'SOMMA', 'TIPO_ENTIDAD_REGIONAL', 'ESTADO_x', 'CodigoInvente', 'DenominacionSocial', 'FormaJuridica_Codigo', 'FormaJuridica_Descripcion', 'NIF', 'codigoDir3', 'codigoOrigen', 'Provincia_Codigo', 'C_ID_UD_ORGANICA', 'C_DNM_UD_ORGANICA', 'C_ID_NIVEL_ADMON', 'C_ID_TIPO_ENT_PUBLICA', 'N_NIVEL_JERARQUICO', 'C_ID_DEP_UD_SUPERIOR', 'C_DNM_UD_ORGANICA_SUPERIOR', 'C_ID_DEP_UD_PRINCIPAL', 'C_DNM_UD_ORGANICA_PRINCIPAL', 'B_SW_DEP_EDP_PRINCIPAL', 'C_ID_DEP_EDP_PRINCIPAL', 'C_DNM_UD_ORGANICA_EDP_PRINCIPAL', 'C_ID_ESTADO', 'D_VIG_ALTA_OFICIAL', 'NIF_CIF', 'C_ID_AMB_PROVINCIA', 'C_DESC_PROV', 'CONTACTOS')

    #Casteamos las columnas correspondientes a valor entero
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('Provincia_Entidad',dataframe_output_unificacion4.Provincia_Entidad.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('ID_ENTIDAD',dataframe_output_unificacion4.ID_ENTIDAD.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('COD_POSTAL',dataframe_output_unificacion4.COD_POSTAL.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('COD_PROVINCIA',dataframe_output_unificacion4.COD_PROVINCIA.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('COD_CCAA',dataframe_output_unificacion4.COD_CCAA.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('Provincia_Codigo',dataframe_output_unificacion4.Provincia_Codigo.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('C_ID_NIVEL_ADMON',dataframe_output_unificacion4.C_ID_NIVEL_ADMON.cast(IntegerType()))
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('N_NIVEL_JERARQUICO',dataframe_output_unificacion4.N_NIVEL_JERARQUICO.cast(IntegerType()))

    #Guardamos los parquets
    dataframe_output_unificacion4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_dataframe_output_unificacion4)
    dataframe_output_unificacion4 = spark.read.parquet(Ruta_Output + df_name_dataframe_output_unificacion4)

    
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    dataframe_output_unificacion4 = dataframe_output_unificacion4.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    #Creamos el fichero pandas
    pd_dataframe_output_unificacion4 = dataframe_output_unificacion4.dropDuplicates().toPandas()

    #Casteamos a enteros los valores necesarios
    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_dataframe_output_unificacion4.columns:
            pd_dataframe_output_unificacion4[column] = np.where(pd_dataframe_output_unificacion4[column]==' ', np.nan, pd_dataframe_output_unificacion4[column])
            pd_dataframe_output_unificacion4[column] = pd_dataframe_output_unificacion4[column].astype('float').astype('Int64')

    #Eliminamos los saltos de carro
    pd_dataframe_output_unificacion4 = pd_dataframe_output_unificacion4.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_dataframe_output_unificacion4.to_csv(Ruta_Output + df_name_dataframe_output_unificacion4 + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(dataframe_output_unificacion4.count())
    dataframe_output_unificacion4.show()






    #Guardamos copia de seguridad de todos los ficheros

    print('dataframe1_output_cruzan_cruce1')
    dataframe1_output_cruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1_pre_cs')

    print('dataframe2_output_nocruzan_cruce1')
    dataframe2_output_nocruzan_cruce1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1_pre_cs')

    print('dataframe1_output_cruzan_cruce2')
    dataframe1_output_cruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2_pre_cs')

    print('dataframe2_output_nocruzan_cruce2')
    dataframe2_output_nocruzan_cruce2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2_pre_cs')

    print('dataframe1_output_cruzan_cruce3')
    dataframe1_output_cruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3_pre_cs')

    print('dataframe2_output_nocruzan_cruce3')
    dataframe2_output_nocruzan_cruce3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3_pre_cs')

    print('dataframe1_output_cruzan_cruce4')
    dataframe1_output_cruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4_pre_cs')

    print('dataframe2_output_nocruzan_cruce4')
    dataframe2_output_nocruzan_cruce4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4_pre_cs')

    print('dataframe1_output_cruzan_cruce5')
    dataframe1_output_cruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5_pre_cs')

    print('dataframe2_output_nocruzan_cruce5')
    dataframe2_output_nocruzan_cruce5.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5_pre_cs')

    print('dataframe1_output_cruzan_cruce6')
    dataframe1_output_cruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6_pre_cs')

    print('dataframe2_output_nocruzan_cruce6')
    dataframe2_output_nocruzan_cruce6.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6_pre_cs')



    print('dataframe_output_unificacion1')
    dataframe_output_unificacion1.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion1_pre_cs')

    print('dataframe_output_unificacion2')
    dataframe_output_unificacion2.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion2_pre_cs')

    print('dataframe_output_unificacion3')
    dataframe_output_unificacion3.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion3_pre_cs')

    print('dataframe_output_unificacion4')
    dataframe_output_unificacion4.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + 'dataframe_output_unificacion4_pre_cs')






    #Leemos copia de seguridad de todos los ficheros

    print('dataframe1_output_cruzan_cruce1')
    dataframe1_output_cruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce1_pre_cs')

    print('dataframe2_output_nocruzan_cruce1')
    dataframe2_output_nocruzan_cruce1 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce1_pre_cs')

    print('dataframe1_output_cruzan_cruce2')
    dataframe1_output_cruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce2_pre_cs')

    print('dataframe2_output_nocruzan_cruce2')
    dataframe2_output_nocruzan_cruce2 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce2_pre_cs')

    print('dataframe1_output_cruzan_cruce3')
    dataframe1_output_cruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce3_pre_cs')

    print('dataframe2_output_nocruzan_cruce3')
    dataframe2_output_nocruzan_cruce3 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce3_pre_cs')

    print('dataframe1_output_cruzan_cruce4')
    dataframe1_output_cruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce4_pre_cs')

    print('dataframe2_output_nocruzan_cruce4')
    dataframe2_output_nocruzan_cruce4 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce4_pre_cs')

    print('dataframe1_output_cruzan_cruce5')
    dataframe1_output_cruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce5_pre_cs')

    print('dataframe2_output_nocruzan_cruce5')
    dataframe2_output_nocruzan_cruce5 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce5_pre_cs')

    print('dataframe1_output_cruzan_cruce6')
    dataframe1_output_cruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe1_output_cruzan_cruce6_pre_cs')

    print('dataframe2_output_nocruzan_cruce6')
    dataframe2_output_nocruzan_cruce6 = spark.read.parquet(Ruta_Output + 'dataframe2_output_nocruzan_cruce6_pre_cs')



    print('dataframe_output_unificacion1')
    dataframe_output_unificacion1 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion1_pre_cs')

    print('dataframe_output_unificacion2')
    dataframe_output_unificacion2 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion2_pre_cs')

    print('dataframe_output_unificacion3')
    dataframe_output_unificacion3 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion3_pre_cs')

    print('dataframe_output_unificacion4')
    dataframe_output_unificacion4 = spark.read.parquet(Ruta_Output + 'dataframe_output_unificacion4_pre_cs')




    #Guardamos la copia de seguridad
    print('Dataframe_nueva_invente_nif')
    Dataframe_nueva_invente_nif.write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_invente_nif_cs')

    print('Dataframe_nueva_DIR3_nif')
    Dataframe_nueva_DIR3_nif.write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_DIR3_nif_cs')

    print('Dataframe_nueva_invente_nombre')
    Dataframe_nueva_invente_nombre.write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_invente_nombre_cs')

    print('Dataframe_nueva_DIR3_nombre')
    Dataframe_nueva_DIR3_nombre.write.mode("overwrite").parquet(Ruta_Output + 'Dataframe_nueva_DIR3_nombre_cs')





    #Leemos la copia de seguridad
    print('Dataframe_nueva_invente_nif')
    Dataframe_nueva_invente_nif = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_invente_nif_cs')

    print('Dataframe_nueva_DIR3_nif')
    Dataframe_nueva_DIR3_nif = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_DIR3_nif_cs')

    print('Dataframe_nueva_invente_nombre')
    Dataframe_nueva_invente_nombre = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_invente_nombre_cs')

    print('Dataframe_nueva_DIR3_nombre')
    Dataframe_nueva_DIR3_nombre = spark.read.parquet(Ruta_Output + 'Dataframe_nueva_DIR3_nombre_cs')











    print(df_name_Dataframe_nueva_invente_nif)

    ##Añadimos los Id de la tabla maestra
    #Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.join(dataframe_output_unificacion5.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF'),
    #                                                         (Dataframe_nueva_invente_nif.Entidad_Norm == dataframe_output_unificacion5.Entidad_Norm) &
    #                                                         (Dataframe_nueva_invente_nif.Provincia_Entidad == dataframe_output_unificacion5.Provincia_Entidad) &
    #                                                         (Dataframe_nueva_invente_nif.CIF == dataframe_output_unificacion5.CIF),
    #                                                         'left').drop(dataframe_output_unificacion5.Entidad_Norm).drop(dataframe_output_unificacion5.Provincia_Entidad).drop(dataframe_output_unificacion5.CIF)

    #Añadimos la información asociada al registro de Invente
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.join(Dataframe_invente_orig,
                                                             (Dataframe_nueva_invente_nif.Entidad_Norm == Dataframe_invente_orig.Entidad_Norm) &
                                                             (Dataframe_nueva_invente_nif.Provincia_Entidad == Dataframe_invente_orig.Provincia_Entidad) &
                                                             (Dataframe_nueva_invente_nif.CIF == Dataframe_invente_orig.CIF),
                                                             'left').drop(Dataframe_invente_orig.Entidad_Norm).drop(Dataframe_invente_orig.Provincia_Entidad).drop(Dataframe_invente_orig.CIF)

    #Casteamos campos a entero
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(None))
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.withColumn('Provincia_Entidad',Dataframe_nueva_invente_nif.Provincia_Entidad.cast(IntegerType()))

    #Guardamos el parquet
    Dataframe_nueva_invente_nif.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_Dataframe_nueva_invente_nif)
    Dataframe_nueva_invente_nif = spark.read.parquet(Ruta_Output + df_name_Dataframe_nueva_invente_nif)

    
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.select('Id', 'CIF', 'Entidad_Norm', 'Provincia_Entidad', 'CodigoInvente', 'DenominacionSocial', 'FormaJuridica_Codigo', 'FormaJuridica_Descripcion', 'NIF', 'codigoDir3', 'codigoOrigen', 'Provincia_Codigo', 'Entidad')

    
    #En el fichero pandas modificamos los formatos necesarios a entero
    pd_Dataframe_nueva_invente_nif = Dataframe_nueva_invente_nif.dropDuplicates().toPandas()

    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_Dataframe_nueva_invente_nif.columns:
            pd_Dataframe_nueva_invente_nif[column] = np.where(pd_Dataframe_nueva_invente_nif[column]==' ', np.nan, pd_Dataframe_nueva_invente_nif[column])
            pd_Dataframe_nueva_invente_nif[column] = pd_Dataframe_nueva_invente_nif[column].astype('float').astype('Int64')

    #En el fichero pandas modificamos los los saltos de carro y guardamos        
    pd_Dataframe_nueva_invente_nif = pd_Dataframe_nueva_invente_nif.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_Dataframe_nueva_invente_nif.to_csv(Ruta_Output + df_name_Dataframe_nueva_invente_nif + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(Dataframe_nueva_invente_nif.count())
    Dataframe_nueva_invente_nif.show()











    print(df_name_Dataframe_nueva_DIR3_nif)

    ##Añadimos los Id de la tabla maestra
    #Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.join(dataframe_output_unificacion5.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF'),
    #                                                         (Dataframe_nueva_DIR3_nif.Entidad_Norm == dataframe_output_unificacion5.Entidad_Norm) &
    #                                                         (Dataframe_nueva_DIR3_nif.Provincia_Entidad == dataframe_output_unificacion5.Provincia_Entidad) &
    #                                                         (Dataframe_nueva_DIR3_nif.CIF == dataframe_output_unificacion5.CIF),
    #                                                         'left').drop(dataframe_output_unificacion5.Entidad_Norm).drop(dataframe_output_unificacion5.Provincia_Entidad).drop(dataframe_output_unificacion5.CIF)

    #Añadimos la información asociada al registro de DIR3
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.join(Dataframe_DIR3_orig,
                                                             (Dataframe_nueva_DIR3_nif.Entidad_Norm == Dataframe_DIR3_orig.Entidad_Norm) &
                                                             (Dataframe_nueva_DIR3_nif.Provincia_Entidad == Dataframe_DIR3_orig.Provincia_Entidad) &
                                                             (Dataframe_nueva_DIR3_nif.CIF == Dataframe_DIR3_orig.CIF),
                                                             'left').drop(Dataframe_DIR3_orig.Entidad_Norm).drop(Dataframe_DIR3_orig.Provincia_Entidad).drop(Dataframe_DIR3_orig.CIF)

    #Casteamos campos a entero
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(None))
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('Provincia_Entidad',Dataframe_nueva_DIR3_nif.Provincia_Entidad.cast(IntegerType()))

    #Guardamos el parquet
    Dataframe_nueva_DIR3_nif.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_Dataframe_nueva_DIR3_nif)
    Dataframe_nueva_DIR3_nif = spark.read.parquet(Ruta_Output + df_name_Dataframe_nueva_DIR3_nif)

    
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    
    Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'C_ID_UD_ORGANICA', 'C_DNM_UD_ORGANICA', 'C_ID_NIVEL_ADMON', 'C_ID_TIPO_ENT_PUBLICA', 'N_NIVEL_JERARQUICO', 'C_ID_DEP_UD_SUPERIOR', 'C_DNM_UD_ORGANICA_SUPERIOR', 'C_ID_DEP_UD_PRINCIPAL', 'C_DNM_UD_ORGANICA_PRINCIPAL', 'B_SW_DEP_EDP_PRINCIPAL', 'C_ID_DEP_EDP_PRINCIPAL', 'C_DNM_UD_ORGANICA_EDP_PRINCIPAL', 'C_ID_ESTADO', 'D_VIG_ALTA_OFICIAL', 'NIF_CIF', 'C_ID_AMB_PROVINCIA', 'C_DESC_PROV', 'CONTACTOS')
    
    #En el fichero pandas modificamos los formatos necesarios a entero
    pd_Dataframe_nueva_DIR3_nif = Dataframe_nueva_DIR3_nif.dropDuplicates().toPandas()

    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_Dataframe_nueva_DIR3_nif.columns:
            pd_Dataframe_nueva_DIR3_nif[column] = np.where(pd_Dataframe_nueva_DIR3_nif[column]==' ', np.nan, pd_Dataframe_nueva_DIR3_nif[column])
            pd_Dataframe_nueva_DIR3_nif[column] = pd_Dataframe_nueva_DIR3_nif[column].astype('float').astype('Int64')

    #En el fichero pandas modificamos los los saltos de carro y guardamos
    pd_Dataframe_nueva_DIR3_nif = pd_Dataframe_nueva_DIR3_nif.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_Dataframe_nueva_DIR3_nif.to_csv(Ruta_Output + df_name_Dataframe_nueva_DIR3_nif + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(Dataframe_nueva_DIR3_nif.count())
    Dataframe_nueva_DIR3_nif.show()











    print(df_name_Dataframe_nueva_invente_nombre)

    ##Añadimos los Id de la tabla maestra
    #Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.join(dataframe_output_unificacion5.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF'),
    #                                                         (Dataframe_nueva_invente_nombre.Entidad_Norm == dataframe_output_unificacion5.Entidad_Norm) &
    #                                                         (Dataframe_nueva_invente_nombre.Provincia_Entidad == dataframe_output_unificacion5.Provincia_Entidad) &
    #                                                         (Dataframe_nueva_invente_nombre.CIF == dataframe_output_unificacion5.CIF),
    #                                                         'left').drop(dataframe_output_unificacion5.Entidad_Norm).drop(dataframe_output_unificacion5.Provincia_Entidad).drop(dataframe_output_unificacion5.CIF)

    #Añadimos la información asociada al registro de Invente
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.join(Dataframe_invente_orig,
                                                                         (Dataframe_nueva_invente_nombre.Entidad_Norm == Dataframe_invente_orig.Entidad_Norm) &
                                                                         (Dataframe_nueva_invente_nombre.Provincia_Entidad == Dataframe_invente_orig.Provincia_Entidad) &
                                                                         (Dataframe_nueva_invente_nombre.CIF == Dataframe_invente_orig.CIF),
                                                                         'left').drop(Dataframe_invente_orig.Entidad_Norm).drop(Dataframe_invente_orig.Provincia_Entidad).drop(Dataframe_invente_orig.CIF)

    #Casteamos campos a entero
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(None))
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.withColumn('Provincia_Entidad',Dataframe_nueva_invente_nombre.Provincia_Entidad.cast(IntegerType()))

    #Guardamos el parquet
    Dataframe_nueva_invente_nombre.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_Dataframe_nueva_invente_nombre)
    Dataframe_nueva_invente_nombre = spark.read.parquet(Ruta_Output + df_name_Dataframe_nueva_invente_nombre)


    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    
    Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'CodigoInvente', 'DenominacionSocial', 'FormaJuridica_Codigo', 'FormaJuridica_Descripcion', 'NIF', 'codigoDir3', 'codigoOrigen', 'Provincia_Codigo')

    
    #En el fichero pandas modificamos los formatos necesarios a entero
    pd_Dataframe_nueva_invente_nombre = Dataframe_nueva_invente_nombre.dropDuplicates().toPandas()

    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_Dataframe_nueva_invente_nombre.columns:
            pd_Dataframe_nueva_invente_nombre[column] = np.where(pd_Dataframe_nueva_invente_nombre[column]==' ', np.nan, pd_Dataframe_nueva_invente_nombre[column])
            pd_Dataframe_nueva_invente_nombre[column] = pd_Dataframe_nueva_invente_nombre[column].astype('float').astype('Int64')

    #En el fichero pandas modificamos los los saltos de carro y guardamos
    pd_Dataframe_nueva_invente_nombre = pd_Dataframe_nueva_invente_nombre.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_Dataframe_nueva_invente_nombre.to_csv(Ruta_Output + df_name_Dataframe_nueva_invente_nombre + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)





    print(Dataframe_nueva_invente_nombre.count())
    Dataframe_nueva_invente_nombre.show()





    print(df_name_Dataframe_nueva_DIR3_nombre)

    ##Añadimos los Id de la tabla maestra
    #Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.join(dataframe_output_unificacion5.select('Id', 'Entidad_Norm', 'Provincia_Entidad', 'CIF'),
    #                                                         (Dataframe_nueva_DIR3_nombre.Entidad_Norm == dataframe_output_unificacion5.Entidad_Norm) &
    #                                                         (Dataframe_nueva_DIR3_nombre.Provincia_Entidad == dataframe_output_unificacion5.Provincia_Entidad) &
    #                                                         (Dataframe_nueva_DIR3_nombre.CIF == dataframe_output_unificacion5.CIF),
    #                                                         'left').drop(dataframe_output_unificacion5.Entidad_Norm).drop(dataframe_output_unificacion5.Provincia_Entidad).drop(dataframe_output_unificacion5.CIF)

    #Añadimos la información asociada al registro de DIR3
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.join(Dataframe_DIR3_orig,
                                                             (Dataframe_nueva_DIR3_nombre.Entidad_Norm == Dataframe_DIR3_orig.Entidad_Norm) &
                                                             (Dataframe_nueva_DIR3_nombre.Provincia_Entidad == Dataframe_DIR3_orig.Provincia_Entidad) &
                                                             (Dataframe_nueva_DIR3_nombre.CIF == Dataframe_DIR3_orig.CIF),
                                                             'left').drop(Dataframe_DIR3_orig.Entidad_Norm).drop(Dataframe_DIR3_orig.Provincia_Entidad).drop(Dataframe_DIR3_orig.CIF)

    #Casteamos campos a entero
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') != 'NaN', F.col('Provincia_Entidad')).otherwise(None))
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('Provincia_Entidad',Dataframe_nueva_DIR3_nombre.Provincia_Entidad.cast(IntegerType()))

    #Guardamos el parquet
    Dataframe_nueva_DIR3_nombre.dropDuplicates().write.mode("overwrite").parquet(Ruta_Output + df_name_Dataframe_nueva_DIR3_nombre)
    Dataframe_nueva_DIR3_nombre = spark.read.parquet(Ruta_Output + df_name_Dataframe_nueva_DIR3_nombre)

    
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('CIF', F.when(F.col('CIF') != ' ', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('CIF', F.when(F.col('CIF') != 'nan', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('CIF', F.when(F.col('CIF') != 'NaN', F.col('CIF')).otherwise(F.lit(None)))
    
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.withColumn('Provincia_Entidad', F.when(F.col('Provincia_Entidad') == ' ', F.col('Provincia_Entidad')).otherwise(F.lit(None)))
    
    
    Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.select('Id', 'CIF', 'Entidad_Norm', 'Entidad', 'Provincia_Entidad', 'C_ID_UD_ORGANICA', 'C_DNM_UD_ORGANICA', 'C_ID_NIVEL_ADMON', 'C_ID_TIPO_ENT_PUBLICA', 'N_NIVEL_JERARQUICO', 'C_ID_DEP_UD_SUPERIOR', 'C_DNM_UD_ORGANICA_SUPERIOR', 'C_ID_DEP_UD_PRINCIPAL', 'C_DNM_UD_ORGANICA_PRINCIPAL', 'B_SW_DEP_EDP_PRINCIPAL', 'C_ID_DEP_EDP_PRINCIPAL', 'C_DNM_UD_ORGANICA_EDP_PRINCIPAL', 'C_ID_ESTADO', 'D_VIG_ALTA_OFICIAL', 'NIF_CIF', 'C_ID_AMB_PROVINCIA', 'C_DESC_PROV', 'CONTACTOS')
    
    #En el fichero pandas modificamos los formatos necesarios a entero
    pd_Dataframe_nueva_DIR3_nombre = Dataframe_nueva_DIR3_nombre.dropDuplicates().toPandas()

    for column in ['Id', 'Provincia_Entidad', 'Provincia_Match', 'ID_ENTIDAD', 'COD_POSTAL', 'COD_PROVINCIA', 'COD_CCAA', 'Provincia_Codigo', 'FormaJuridica_Codigo', 'C_ID_NIVEL_ADMON', 'N_NIVEL_JERARQUICO']:
        if column in pd_Dataframe_nueva_DIR3_nombre.columns:
            pd_Dataframe_nueva_DIR3_nombre[column] = np.where(pd_Dataframe_nueva_DIR3_nombre[column]==' ', np.nan, pd_Dataframe_nueva_DIR3_nombre[column])
            pd_Dataframe_nueva_DIR3_nombre[column] = pd_Dataframe_nueva_DIR3_nombre[column].astype('float').astype('Int64')

    #En el fichero pandas modificamos los los saltos de carro y guardamos
    pd_Dataframe_nueva_DIR3_nombre = pd_Dataframe_nueva_DIR3_nombre.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)

    #Guardamos el csv
    pd_Dataframe_nueva_DIR3_nombre.to_csv(Ruta_Output + df_name_Dataframe_nueva_DIR3_nombre + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    print(Dataframe_nueva_DIR3_nombre.count())
    Dataframe_nueva_DIR3_nombre.show()
    


    
    
    
    
    
    
    
    
    NIFs_coinciden_Altas_nuevas = pd.read_csv(Ruta_Output + 'NIFs_coinciden_Altas_nuevas' + '.csv', sep=";", quotechar='"')

    #Creamos el esquema para el fichero pd_dataframe_output_unificacion5
    schema = StructType([
        StructField('source_names', StringType(), True),
        StructField('CIF', StringType(), True),
        StructField('source_municipality', StringType(), True),
        StructField('target_names', StringType(), True),
        StructField('target_municipality', StringType(), True)])

    #Creamos el fichero dataframe_output_unificacion5
    NIFs_coinciden_Altas_nuevas = spark.createDataFrame(NIFs_coinciden_Altas_nuevas, schema)


    results = NIFs_coinciden_Altas_nuevas

    stopwords_es = nltk.corpus.stopwords.words('spanish')


    results = results.filter((F.col('source_names') != F.col('target_names')) |
                             (F.col('source_municipality') != F.col('target_municipality')))

    results = results.withColumn("aux", F.split("source_names", "\\s+"))
    remover = StopWordsRemover(stopWords=stopwords_es, inputCol="aux", outputCol="source_names_stopwords")
    results = remover.transform(results).withColumn("source_names_stopwords", F.array_join("source_names_stopwords", " "))

    results = results.withColumn("aux", F.split("target_names", "\\s+"))
    remover = StopWordsRemover(stopWords=stopwords_es, inputCol="aux", outputCol="target_names_stopwords")
    results = remover.transform(results).withColumn("target_names_stopwords", F.array_join("target_names_stopwords", " "))


    #return spark.createDataFrame(results, schema).withColumn('final_score', 0.9*(udf_Distance_ratcliff_obershelp(F.col('source_names'), F.col('target_names'))) + F.col('weighted_city_score'))
    results =  results.withColumn('final_score', udf_Distance_ratcliff_obershelp(F.col('source_names_stopwords'), F.col('target_names_stopwords'))).withColumn('final_score', F.when(F.col('source_names_stopwords') != (F.col('target_names_stopwords')), F.col('final_score')).otherwise(F.lit(1))).drop('aux').drop('source_names_stopwords').drop('target_names_stopwords').drop_duplicates()


    results = results.dropDuplicates().toPandas()

    #Casteamos las columnas
    for column in ['Id_source', 'Provincia_Entidad_source', 'Id_target', 'Provincia_Entidad_target']:
        if column in results.columns:
            results[column] = np.where(results[column]==' ', np.nan, results[column])
            results[column] = results[column].astype('float').astype('Int64')

    #Guardamos el fichero en csv
    results.to_csv(Ruta_Output + 'NIFs_coinciden_Altas_nuevas_Distance' + '.csv', index=False, decimal=',', sep=';', float_format='%.4f', quoting=csv.QUOTE_NONNUMERIC)
    
    
    
#Modificaciones, eliminar cruce con ID_match y mach solicitudes la parte de los nombres, y en ambos casos mantener la de los CIFS.