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
    ids_source=[]
    ids_target=[]
    cities_source=[]
    cities_target=[]
    mun_source=[]
    mun_target=[]
    countries_source=[]
    countries_target=[]
    for i in results:
        source_name = results[i]['source_name']
        source_id = results[i]['source_id']  
        source_city = results[i]['source_city']
        source_mun = results[i]['source_municipality']
        source_country = results[i]['source_country']
        for element in results[i]['matching_names_by_elasticsearch']:
            target_name = element[1]
            target_id = element[2]
            target_city = element[3]
            target_mun = element[4]
            target_country = element[5]
            names_source.append(source_name)
            ids_source.append(source_id)
            cities_source.append(source_city)
            mun_source.append(source_mun)
            countries_source.append(source_country)
            names_target.append(target_name)
            ids_target.append(target_id)
            cities_target.append(target_city)
            mun_target.append(target_mun)
            countries_target.append(target_country)
            
    result = pd.DataFrame({'source_name':names_source,
             'target_name':names_target,
             'source_id':ids_source,
             'target_id':ids_target,
             'source_city':cities_source,
             'target_city':cities_target,
             'source_mun':mun_source,
             'target_mun':mun_target,
             'source_country':countries_source,
             'target_country':countries_target})
            
    return result

def compute_scores(cities, syn_cities, results,name_weight,city_weight,mun_weight,country_weight):
    results['name_score'] = results.apply(lambda x: compare_names(x.source_name, x.target_name, cities, syn_cities), axis=1)
    results['id_score'] = results.apply(lambda x: compare_loc(x.source_id, x.target_id), axis=1)
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
                                           id_column_source,
                                           city_column_source,
                                           municipality_column_source,
                                           country_column_source,
                                           df2,
                                           target_names_column,
                                           id_column_target,
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
    ids_source=list(df1[id_column_source])
    ids_target=list(df2[id_column_target])
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
            matching_entities.append((target_idx,entity_j,ids_target[target_idx],cities_target[target_idx],mun_target[target_idx],countries_target[target_idx]))
            
        results[source_idx] = {'source_name':entity_i,
                               'source_id' :ids_source[source_idx],
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
                                                                 id_column_source,
                                                                 city_column_source,
                                                                 source_municipality,
                                                                 country_column_source,
                                                                 df2,
                                                                 target_names,
                                                                 id_column_target,
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
    df_1 = df1.select(source_names, id_column_source, city_column_source).toPandas()
    df_2 = df2.select(target_names, id_column_target, city_column_target).toPandas()
    for column in ['source_municipality', 'source_country']:
        df_1[column] = 0
    for column in ['target_municipality', 'target_country']:
        df_2[column] = 0
        
    #print('df_1')
    #print(df_1)
    #print('df_1')
    #print(df_2)
    #XXXXXX
    results = get_matching_entities_by_elasticSearch(spark,
                                                     syn_cities,
                                                     cities,
                                                     df_1,
                                                     source_names,
                                                     id_column_source,
                                                     city_column_source,
                                                     'source_municipality',
                                                     'source_country',
                                                     df_2,
                                                     target_names,
                                                     id_column_target,
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
        StructField('source_id', StringType(), True), #Le cambio el orden
        StructField('target_id', StringType(), True), #Le cambio el orden
        StructField('source_city', StringType(), True), #Le cambio el orden
        StructField('target_city', StringType(), True), #Le cambio el orden
        StructField('source_country', StringType(), True),                     
        StructField('target_country', StringType(), True),
        StructField('name_score', StringType(), True),
        StructField('id_score', StringType(), True),
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
    
    
    
    
    
    