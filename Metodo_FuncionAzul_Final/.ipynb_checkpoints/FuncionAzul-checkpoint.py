from function2 import *
from fa import *
import pickle
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.window import Window as W


def FuncionAzul(Ruta_Nombre_Input_entidades, Ruta_Nombre_Input_investigadores, 
                Ruta_Nombre_Input_relaciones,
                Ruta_Nombre_Input_invente, Ruta_Nombre_Input_dir3, 
                Ruta_Nombre_Intermedio_cruce1_relaciones,
                Ruta_Nombre_Intermedio_cruce1_entidades, 
                Ruta_Nombre_Intermedio_cruce2_invente, 
                Ruta_Nombre_Intermedio_cruce3_dir3,
                Ruta_Nombre_Output_investigadores, 
                Ruta_Nombre_Output_relaciones,
                Ruta_Nombre_Output_entidades, 
                flag_invente_dir3, 
                Flag_csv):
    '''
    Función que realiza todos los procesos y cruces del flujo 1 y 2 de la Función azul

    Parameters
    ----------
    Ruta_Nombre_Input_entidades : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_ENTIDADES_PUBLICAS_FIL_INVEST.csv.
    Ruta_Nombre_Input_investigadores : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_INVEST_UO_FIL_PB_J_MAESTRO.csv.
    Ruta_Nombre_Input_relaciones : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_RELACIONES_ENTIDADES_MST_AYUDAS.csv.
    Ruta_Nombre_Input_invente : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_IGAE_INVENTE.csv.
    Ruta_Nombre_Input_dir3 : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_DIR3.csv.
    Ruta_Nombre_Intermedio_cruce1_relaciones : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_NV_RELACIONES_MST.csv
    Ruta_Nombre_Intermedio_cruce1_entidades : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_ENTIDADES_CENTROS_PB.csv
    Ruta_Nombre_Intermedio_cruce2_invente : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_ENTIDADES_CENTROS_PB_INVENTE_NORM.csv
    Ruta_Nombre_Intermedio_cruce3_dir3 : STRING
        Dirección del archivo correspondiente en csv. Ejemplo DATA_SIC_ENTIDADES_CENTROS_PB_INVENTE_DIR3_NORM.csv
    Ruta_Nombre_Output_investigadores : STRING
        Dirección del archivo correspondiente en csv.Ejemplo DATA_SIC_INVEST_UO_FIL_PB_CENTRO_MST.csv
    Ruta_Nombre_Output_relaciones : STRING
        DATA_SIC_RELACIONES_ENTIDADES_PB_FIL_INVEST.csv.
    Ruta_Nombre_Output_entidades : STRING
        DATA_SIC_ENTIDADES_PUBLICAS_FIL_INVEST.csv
    flag_invente_dir3 : BOOLEAN
        Si el valor es True, se crean diferentes cruces entre los archivos, completando el flujo 1.
        Si es False se procesa el flujo 2.
    Flag_csv : BOOLEAN
        Cuando sea True se generan los archivos en los formatos csv y parquet. 
        Cuando sea False, se generan los archivos únicamente en formato parquet.
    '''
    spark = SparkSession.builder.appName('FuncionAzul').config("spark.driver.memory","70G")\
                         .config("spark.executor.memory","70G")\
                         .config("spark.executor.cores","20")\
                         .config("spark.executor.instances","5")\
                         .config("spark.driver.maxResultSize", '128g')\
                         .config("spark.memory.offHeap.enabled", 'true')\
                         .config("spark.memory.offHeap.size", '30g')\
                         .enableHiveSupport().getOrCreate()


    data,Nombre_Input_relaciones, Nombre_Input_investigadores, Nombre_Intermedio_cruce1_entidades, Nombre_Output_investigadores = cruce1_entidades_Output_investigadores(spark,
                                                                                                                                                       Ruta_Nombre_Input_entidades,
                                                                                                                                                       Ruta_Nombre_Input_investigadores,
                                                                                                                                                       Ruta_Nombre_Input_relaciones,
                                                                                                                                                       Ruta_Nombre_Intermedio_cruce1_entidades,
                                                                                                                                                       Ruta_Nombre_Output_investigadores, 
                                                                                                                                                       Flag_csv = True)


    Nombre_Intermedio_cruce1_relaciones = cruce1_relaciones(spark,
                                                              data,
                                                              Nombre_Input_relaciones,
                                                              Nombre_Input_investigadores,
                                                              Ruta_Nombre_Intermedio_cruce1_relaciones,
                                                              Flag_csv)


    Nombre_Output_relaciones = output_relaciones(spark,
                                                  Nombre_Input_relaciones,
                                                  Nombre_Intermedio_cruce1_relaciones,
                                                  Ruta_Nombre_Output_relaciones,
                                                  Flag_csv)
    
    if flag_invente_dir3 == True:
        Nombre_Intermedio_cruce2_invente= cruce2_invente(spark,
                                                        Nombre_Input_entidades,
                                                        Ruta_Nombre_Input_invente,
                                                        Nombre_Intermedio_cruce1_entidades,
                                                        Ruta_Nombre_Intermedio_cruce2_invente,
                                                        Ruta_Nombre_Output_entidades,
                                                        Flag_csv)
        Nombre_Intermedio_cruce3_dir3 = cruce3_dir3(spark,
                                                    Nombre_Intermedio_cruce2_invente,
                                                    Ruta_Nombre_Input_dir3,
                                                    Ruta_Nombre_Intermedio_cruce3_dir3,
                                                    Flag_csv)
        Output_entidades_true(spark,
                              Nombre_Input_entidades,
                              Nombre_Intermedio_cruce3_dir3,
                              Ruta_Nombre_Output_entidades)
    else:
        Output_entidades_false(spark,
                              Nombre_Input_entidades,
                              Nombre_Intermedio_cruce1_entidades,
                              Ruta_Nombre_Output_entidades)
    