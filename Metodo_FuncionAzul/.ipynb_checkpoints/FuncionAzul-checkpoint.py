from function2 import *
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
                Ruta_Nombre_Output_entidades, flag_invente_dir3, Flag_csv):
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


    # =============================================================================
    # Creamos 3 archivos: 
    # 1. Nombre_Intermedio_cruce1_entidades.
    # 2. Nombre_Output_investigadores.  
    # 3. Nombre_Intermedio_cruce1_relaciones.
    # =============================================================================
    Nombre_Input_entidades = spark.read.csv(Ruta_Nombre_Input_entidades, sep=";", header=True)
    Nombre_Input_relaciones = spark.read.csv(Ruta_Nombre_Input_relaciones, sep=";", header=True)
    Nombre_Input_investigadores = spark.read.csv(Ruta_Nombre_Input_investigadores, sep=";", header=True)



    cruce1 = Nombre_Input_investigadores.\
                                        join(Nombre_Input_relaciones,
                                             Nombre_Input_investigadores.Id ==  Nombre_Input_relaciones.Id_NIVEL_0,"left")\
                                        .dropDuplicates(Nombre_Input_investigadores.columns)\
                                        .filter("Id_NIVEL_0 is not null")



    cruce1 = cruce1.withColumn("Similitud", 
                               udf_Distance_ratcliff_obershelp(
                                   UDF_normalizarTexto(cruce1.NOMBRE_ENTIDAD_NIVEL_1),
                                   cruce1.Centro_Norm )
                              ) 


    # # Nombre_Intermedio_cruce1_entidades

    # Valor maximo del index del archivo Nombre_Input_entidades
    maxId = Nombre_Input_entidades.withColumn("Id", Nombre_Input_entidades.Id.cast('int')).select("Id").rdd.max()[0]




    data = cruce1.filter("Similitud <0.875")\
                 .withColumn("CIF",lit(None).cast(StringType()))\
                 .withColumn("Entidad_Norm",col("Centro_Norm"))\
                 .withColumn("Provincia_Entidad",col("Provincia_Centro"))\
                 .withColumn("ID_ENTIDAD",lit(None).cast(StringType()))\
                 .withColumn("NIF_COD",lit(None).cast(StringType()))\
                 .withColumn("ACRONIMO",lit(None).cast(StringType()))\
                 .withColumn("NOMBRE_ENTIDAD",lit(None).cast(StringType()))\
                 .withColumn("Nombre_Entidad_Mostrar",lit(None).cast(StringType()))\
                 .withColumn("TIPO_ENTIDAD_N1_1",lit(None).cast(StringType()))\
                 .withColumn("TIPO_ENTIDAD_N2_1",lit(None).cast(StringType()))\
                 .withColumn("DIRECCION_POSTAL",lit(None).cast(StringType()))\
                 .withColumn("COD_POSTAL",lit(None).cast(StringType()))\
                 .withColumn("COD_PROVINCIA",lit(None).cast(StringType()))\
                 .withColumn("PROVINCIA",lit(None).cast(StringType()))\
                 .withColumn("COD_CCAA",lit(None).cast(StringType()))\
                 .withColumn("CCAA",lit(None).cast(StringType()))\
                 .withColumn("ENLACE_WEB",lit(None).cast(StringType()))\
                 .withColumn("SOMMA",lit(None).cast(StringType()))\
                 .withColumn("TIPO_ENTIDAD_REGIONAL",lit(None).cast(StringType()))\
                 .withColumn("ESTADO_x",lit(None).cast(StringType()))\
                 .withColumn("CodigoInvente",lit(None).cast(StringType()))\
                 .withColumn("DenominacionSocial",lit(None).cast(StringType()))\
                 .withColumn("FormaJuridica_Codigo",lit(None).cast(StringType()))\
                 .withColumn("FormaJuridica_Descripcion",lit(None).cast(StringType()))\
                 .withColumn("NIF",lit(None).cast(StringType()))\
                 .withColumn("codigoDir3",lit(None).cast(StringType()))\
                 .withColumn("codigoOrigen",lit(None).cast(StringType()))\
                 .withColumn("Provincia_Codigo",lit(None).cast(StringType()))\
                 .withColumn("C_ID_UD_ORGANICA",lit(None).cast(StringType()))\
                 .withColumn("C_DNM_UD_ORGANICA",lit(None).cast(StringType()))\
                 .withColumn("C_ID_NIVEL_ADMON",lit(None).cast(StringType()))\
                 .withColumn("C_ID_TIPO_ENT_PUBLICA",lit(None).cast(StringType()))\
                 .withColumn("N_NIVEL_JERARQUICO",lit(None).cast(StringType()))\
                 .withColumn("C_ID_DEP_UD_SUPERIOR",lit(None).cast(StringType()))\
                 .withColumn("C_DNM_UD_ORGANICA_SUPERIOR",lit(None).cast(StringType()))\
                 .withColumn("C_ID_DEP_UD_PRINCIPAL",lit(None).cast(StringType()))\
                 .withColumn("C_DNM_UD_ORGANICA_PRINCIPAL",lit(None).cast(StringType()))\
                 .withColumn("B_SW_DEP_EDP_PRINCIPAL",lit(None).cast(StringType()))\
                 .withColumn("C_ID_DEP_EDP_PRINCIPAL",lit(None).cast(StringType()))\
                 .withColumn("C_DNM_UD_ORGANICA_EDP_PRINCIPAL",lit(None).cast(StringType()))\
                 .withColumn("C_ID_ESTADO",lit(None).cast(StringType()))\
                 .withColumn("D_VIG_ALTA_OFICIAL",lit(None).cast(StringType()))\
                 .withColumn("NIF_CIF",lit(None).cast(StringType()))\
                 .withColumn("C_ID_AMB_PROVINCIA",lit(None).cast(StringType()))\
                 .withColumn("C_DESC_PROV",lit(None).cast(StringType()))\
                 .withColumn("CONTACTOS",lit(None).cast(StringType()))\
                 .withColumn("List_Entidad_Norm",col("Centro_Norm"))\
                 .withColumn("List_Provincia_Entidad",col("Provincia_Centro"))\
                 .withColumn("List_CIF",lit(None).cast(StringType()))



    data_count = data.count()

    # =============================================================================
    # Procesamiento para iniciar el archivo deseado a partter del maxId
    # =============================================================================
    data = data.dropDuplicates(["Entidad_Norm", "Provincia_Entidad"])
    data = data.withColumnRenamed("Id", "IdOld")
    
    Index = [maxId + i for i in range(1, data_count+1)]
    Index = spark.createDataFrame(Index, IntegerType())
    Index = Index.withColumnRenamed("value", "Id")




    windowSpec = W.orderBy("Dummy")
    
    Index = Index.withColumn("Dummy", monotonically_increasing_id())
    Index = Index.withColumn("Dummy", F.row_number().over(windowSpec))
    
    data = data.withColumn("Dummy", monotonically_increasing_id())
    data = data.withColumn("Dummy", F.row_number().over(windowSpec))
    
    data = data.join(Index, "Dummy", "outer").drop("Dummy")



    Nombre_Intermedio_cruce1_entidades = data.select(Nombre_Input_entidades.columns)




    
    # =============================================================================
    # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
    # =============================================================================
    save_csv_parquet(Nombre_Intermedio_cruce1_entidades, Flag_csv, Ruta_Nombre_Intermedio_cruce1_entidades)
    
    
    # # Creación del archivo Nombre_Output_investigadores


    data_2 = cruce1.filter("Similitud >=0.875")\
            .withColumn("Final_Score", col("Similitud"))\
            .withColumn("Id_Centro", col("Id_NIVEL_1"))\
            .withColumn("Centro_Nombre_match", col("NOMBRE_ENTIDAD_NIVEL_1"))\
            .withColumn("Centro_Provincia_match", col("COD_PROVINCIA_NIVEL_1"))\
            .withColumn("Similitud_Centro", col("Similitud"))\
            .select(Nombre_Input_investigadores.columns + 
                    ["Final_Score","Id_Centro", "Centro_Nombre_match", "Centro_Provincia_match", "Similitud_Centro"])
    
    data_3 = data.withColumn("Final_Score", col("Similitud"))\
            .withColumn("Id_Centro", col("Id"))\
            .withColumn("Centro_Nombre_match", col("NOMBRE_ENTIDAD_NIVEL_1"))\
            .withColumn("Centro_Provincia_match", col("COD_PROVINCIA_NIVEL_1"))\
            .withColumn("Similitud_Centro", col("Similitud"))\
            .select(Nombre_Input_investigadores.columns + 
                    ["Final_Score","Id_Centro", "Centro_Nombre_match", "Centro_Provincia_match", "Similitud_Centro"])


    Nombre_Output_investigadores = data_2.union(data_3)

    # =============================================================================
    # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
    # =============================================================================
    save_csv_parquet(Nombre_Output_investigadores, Flag_csv, Ruta_Nombre_Output_investigadores)
    


    # =============================================================================
    # Creación del archivo Nombre_Intermedio_cruce1_relaciones
    # =============================================================================
        
    
    Nombre_Intermedio_cruce1_relaciones = data.select(["Id", "IdOld", "Entidad", "Centro","Centro_Norm", "Provincia_Entidad", "Provincia_Centro","CIF","Entidad_Norm"])
    

    #Eliminamos duplicados
    
    l = get_matching_entities_by_elasticSearch(
                                            Nombre_Input_investigadores, 
                                            "Entidad_Norm", 
                                            ["Entidad_Norm"],
                                            Nombre_Input_investigadores, 
                                            "Centro_Norm",
                                            ["Centro_Norm"],
                                            'index_name',
                                            5)

    l = get_matching_by_elasticSearch_Distance_ratcliff_obershelp(spark,l)
    duplicados = list(l.filter("final_score >= 0.875").select("Entidad_Norm").dropDuplicates().toPandas().Entidad_Norm)
    Nombre_Intermedio_cruce1_relaciones = Nombre_Intermedio_cruce1_relaciones.filter(~Nombre_Intermedio_cruce1_relaciones.Entidad_Norm.isin(duplicados))
    
    
    Nombre_Intermedio_cruce1_relaciones = Nombre_Intermedio_cruce1_relaciones.withColumn("Id_NIVEL_1", col("Id"))\
          .withColumn("Id_NIVEL_0", col("IdOld"))\
          .withColumn("NOMBRE_ENTIDAD_NIVEL_0", col("Entidad"))\
          .withColumn("NOMBRE_ENTIDAD_NIVEL_1", col("Centro"))\
          .withColumn("COD_PROVINCIA_NIVEL_0", col("Provincia_Entidad"))\
          .withColumn("COD_PROVINCIA_NIVEL_1", col("Provincia_Centro"))\
          .withColumn("NIF_COD_NIVEL_0", col("CIF"))\
          .withColumn("Jerarquia", lit("Orgánica Directa").cast(StringType()))\
          .withColumn("ACRONIMO_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("ACRONIMO_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("CCAA_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("CCAA_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("COD_CCAA_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("COD_CCAA_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("COD_POSTAL_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("COD_POSTAL_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("DIRECCION_POSTAL_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("DIRECCION_POSTAL_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("ENLACE_WEB_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("ENLACE_WEB_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("ESTADO_x_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("ENLACE_WEB_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("ESTADO_x_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("ESTADO_x_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("ID_ENTIDAD_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("ID_ENTIDAD_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("NIF_COD_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("Nombre_Entidad_Mostrar_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("Nombre_Entidad_Mostrar_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("PROVINCIA_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("PROVINCIA_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("SOMMA_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("SOMMA_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("TIPO_ENTIDAD_N1_1_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("TIPO_ENTIDAD_N1_1_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("TIPO_ENTIDAD_N2_1_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("TIPO_ENTIDAD_N2_1_NIVEL_1",lit(None).cast(StringType()))\
          .withColumn("TIPO_ENTIDAD_REGIONAL_NIVEL_0",lit(None).cast(StringType()))\
          .withColumn("TIPO_ENTIDAD_REGIONAL_NIVEL_1",lit(None).cast(StringType()))\
        



    Nombre_Intermedio_cruce1_relaciones = Nombre_Intermedio_cruce1_relaciones.select(Nombre_Input_relaciones.columns)

    # =============================================================================
    # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
    # =============================================================================
    save_csv_parquet(Nombre_Intermedio_cruce1_relaciones, Flag_csv, Ruta_Nombre_Intermedio_cruce1_relaciones)


    # =============================================================================
    # Agregacion vertical de relaciones
    # =============================================================================

    Nombre_Output_relaciones = Nombre_Input_relaciones.union(Nombre_Intermedio_cruce1_relaciones)
    
    # =============================================================================
    # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
    # =============================================================================
    save_csv_parquet(Nombre_Output_relaciones, Flag_csv, Ruta_Nombre_Output_relaciones)
    

    if flag_invente_dir3 == True:
    # # Nombre_Input_invente
    # data invente
        Nombre_Input_invente = pd.read_excel(Ruta_Nombre_Input_invente)
        for i in Nombre_Input_invente.columns:
            Nombre_Input_invente[i] = Nombre_Input_invente[i].astype(str)
        


        new_header = Nombre_Input_invente.iloc[0] 
        # #grab the first row for the header
        Nombre_Input_invente = Nombre_Input_invente[1:] #take the data less the header row
        Nombre_Input_invente.columns = new_header #set the header row as the df header
        
        Nombre_Input_invente = Nombre_Input_invente[["DenominacionSocial", "Provincia_Codigo", "NIF",
                                                        "CodigoInvente", "FormaJuridica_Codigo", 
                                                        "FormaJuridica_Descripcion", "codigoDir3","codigoOrigen",
                                                        "Provincia_Codigo"]]
        
        Nombre_Input_invente = Nombre_Input_invente.loc[:,~Nombre_Input_invente.columns.duplicated()]
        
        Nombre_Input_invente = spark.createDataFrame(Nombre_Input_invente.fillna('')) 
        
        
        Nombre_Input_invente = Nombre_Input_invente.withColumn("DenominacionSocial_Norm", UDF_normalizarTexto(Nombre_Input_invente.DenominacionSocial))


        # # Nombre_Intermedio_cruce2_invente
        
        
        
        l = get_matching_entities_by_elasticSearch(
                                                Nombre_Intermedio_cruce1_entidades, 
                                                "Entidad_Norm", 
                                                ["Entidad_Norm"],
                                                Nombre_Input_invente, 
                                                "DenominacionSocial_Norm",
                                                Nombre_Input_invente.columns,
                                                'index_name',
                                                1)



        l = get_matching_by_elasticSearch_Distance_ratcliff_obershelp(spark,l).filter("final_score >= 0.875")
        
        
        
        l = l.sort(l.final_score.desc()).dropDuplicates(subset = ['Entidad_Norm']).select("Entidad_Norm","DenominacionSocial", "NIF",
                                                                                        "CodigoInvente", "FormaJuridica_Codigo", 
                                                                                        "FormaJuridica_Descripcion", "codigoDir3","codigoOrigen",
                                                                                        "Provincia_Codigo", "final_score","DenominacionSocial_Norm")
        


        Nombre_Intermedio_cruce2_invente = Nombre_Intermedio_cruce1_entidades.drop("CodigoInvente","DenominacionSocial", "FormaJuridica_Codigo", 
                                                                           "FormaJuridica_Descripcion", "NIF", "codigoDir3", "codigoOrigen",
                                                                            "Provincia_Codigo", "Provincia_Entidad", "CIF")







        Nombre_Intermedio_cruce2_invente = Nombre_Intermedio_cruce2_invente.join(l,
                                              Nombre_Intermedio_cruce2_invente.Entidad_Norm ==  l.Entidad_Norm,"left")                                 
        
           
        
        Nombre_Intermedio_cruce2_invente1 = Nombre_Intermedio_cruce2_invente.filter("final_score is not NULL").drop("Entidad_Norm")
        Nombre_Intermedio_cruce2_invente2 = Nombre_Intermedio_cruce2_invente.filter("final_score is  NULL")
        
        
        
        
        Nombre_Intermedio_cruce2_invente1 = Nombre_Intermedio_cruce2_invente1.withColumn("Entidad_Norm", col("DenominacionSocial_Norm"))\
                                                                            .withColumn("Provincia_Entidad", col("Provincia_Codigo"))\
                                                                            .withColumn("CIF", col("NIF"))\
                                                                            .withColumn("Similitud_invente", col("final_score"))
        
        


        Nombre_Intermedio_cruce2_invente1 = Nombre_Intermedio_cruce2_invente1.select(Nombre_Intermedio_cruce1_entidades.columns+["Similitud_invente"])




        Nombre_Intermedio_cruce2_invente = Nombre_Intermedio_cruce2_invente1.union(Nombre_Intermedio_cruce2_invente2)
        
        # =============================================================================
        # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
        # =============================================================================
        save_csv_parquet(Nombre_Intermedio_cruce2_invente, Flag_csv, Ruta_Nombre_Intermedio_cruce2_invente)



        # # Nombre_Intermedio_proceso_dir3_entidades
        
        
        
        Nombre_Input_dir3 = spark.read.csv(Ruta_Nombre_Input_dir3, sep=";", header=True).drop("Unnamed: 0")
        Nombre_Input_dir3 = Nombre_Input_dir3.withColumn("C_DNM_UD_ORGANICA_Norm", UDF_normalizarTexto(Nombre_Input_dir3.C_DNM_UD_ORGANICA))




        l = get_matching_entities_by_elasticSearch(
                                                Nombre_Intermedio_cruce2_invente, 
                                                "Entidad_Norm", 
                                                ["Entidad_Norm"],
                                                Nombre_Input_dir3, 
                                                "C_DNM_UD_ORGANICA_Norm",
                                                Nombre_Input_dir3.columns,
                                                'index_name',
                                                5)

        l = get_matching_by_elasticSearch_Distance_ratcliff_obershelp(spark,l).filter("final_score >= 0.875")\
                                       .dropDuplicates(subset = ['Entidad_Norm', "C_DNM_UD_ORGANICA_Norm"])
                                                                            
        



        unique = l.selectExpr(
                      '*', 
                      'count(*) over (partition by Entidad_Norm) as cnt'
                    ).filter(F.col('cnt') == 1).drop('cnt')




        duplicated = l.join(l.groupBy('Entidad_Norm')\
                          .count().where('count = 1').drop('count'),
                        on=['Entidad_Norm'],
                        how='left_anti')



        duplicated1 = duplicated.selectExpr(
                      '*', 
                      'count(*) over (partition by Entidad_Norm, N_NIVEL_JERARQUICO) as cnt'
                    ).filter(F.col('cnt') == 1).drop('cnt')

        duplicated1 = duplicated1.sort(duplicated1.N_NIVEL_JERARQUICO.asc())\
                                    .dropDuplicates(subset = ['Entidad_Norm'])
        
        
        
        duplicated2 = l.join(l.groupBy('Entidad_Norm', 'N_NIVEL_JERARQUICO')\
                          .count().where('count = 1').drop('count'),
                        on=['Entidad_Norm','N_NIVEL_JERARQUICO'],
                        how='left_anti')


        duplicated2 = duplicated2.withColumn("Similitud", 
                                   udf_Distance_ratcliff_obershelp(
                                       UDF_normalizarTexto(duplicated2.C_DNM_UD_ORGANICA_SUPERIOR),
                                       duplicated2.Entidad_Norm )
                                  ) 



        duplicated2 = duplicated2.sort(duplicated2.Similitud.desc())\
                                    .dropDuplicates(subset = ['Entidad_Norm']).drop("Similitud")
        
        
        
        match = unique.union(duplicated1).union(duplicated2)\
                       .select(['Entidad_Norm',"final_score"] + Nombre_Input_dir3.columns)\
                       .withColumnRenamed("Entidad_Norm","Entidad_Norm1")
        


        Nombre_Intermedio_proceso_dir3_entidades = Nombre_Intermedio_cruce2_invente\
                                                    .drop('C_DNM_UD_ORGANICA_SUPERIOR', 'C_ID_UD_ORGANICA', 'C_ID_TIPO_ENT_PUBLICA',
                                                          'C_DESC_PROV', 'C_ID_DEP_EDP_PRINCIPAL', 'C_ID_ESTADO', 'D_VIG_ALTA_OFICIAL', 
                                                          'CONTACTOS', 'NIF_CIF', 'C_ID_NIVEL_ADMON', 'C_DNM_UD_ORGANICA_EDP_PRINCIPAL', 
                                                          'C_ID_DEP_UD_PRINCIPAL', 'N_NIVEL_JERARQUICO', 'C_DNM_UD_ORGANICA', 
                                                          'C_DNM_UD_ORGANICA_PRINCIPAL', 'B_SW_DEP_EDP_PRINCIPAL', 'C_ID_AMB_PROVINCIA',
                                                          'C_ID_DEP_UD_SUPERIOR')




        Nombre_Intermedio_proceso_dir3_entidades = Nombre_Intermedio_proceso_dir3_entidades.join(match,
                                              Nombre_Intermedio_proceso_dir3_entidades.Entidad_Norm ==  match.Entidad_Norm1,"left")     
        
        
        
        
        Nombre_Intermedio_proceso_dir3_entidades1 = Nombre_Intermedio_proceso_dir3_entidades.filter("final_score is not NULL").drop("Entidad_Norm")
        
        Nombre_Intermedio_proceso_dir3_entidades2 = Nombre_Intermedio_proceso_dir3_entidades.filter("final_score is  NULL").drop("Entidad_Norm1")
        
        Nombre_Intermedio_proceso_dir3_entidades2 = Nombre_Intermedio_proceso_dir3_entidades2\
                                                        .withColumn("Similitud_Dir3",lit(None).cast(FloatType()))



        Nombre_Intermedio_proceso_dir3_entidades1 = Nombre_Intermedio_proceso_dir3_entidades1\
                                        .withColumn("Entidad_Norm", col("C_DNM_UD_ORGANICA"))\
                                        .withColumn("Provincia_Entidad", col("C_DESC_PROV"))\
                                        .withColumn("CIF", col("NIF_CIF"))\
                                        .withColumn("Similitud_Dir3", col("final_score"))\
                                        .drop("Entidad_Norm1")




        Nombre_Intermedio_cruce3_dir3 = Nombre_Intermedio_proceso_dir3_entidades1\
                                                    .union(Nombre_Intermedio_proceso_dir3_entidades2)\
                                                    .drop("C_DNM_UD_ORGANICA_Norm", "final_score")
        
        # =============================================================================
        # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
        # =============================================================================
        save_csv_parquet(Nombre_Intermedio_cruce3_dir3, Flag_csv, Ruta_Nombre_Intermedio_cruce3_dir3)


        # # Inclusion vertical
        
        
        
        Nombre_Output_entidades = Nombre_Intermedio_cruce3_dir3.drop("Similitud_Invente", "Similitud_Dir3")
        
        
        
        
        Nombre_Output_entidades = Nombre_Input_entidades\
                                                    .union(Nombre_Output_entidades)

        
        # =============================================================================
        # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
        # =============================================================================
        save_csv_parquet(Nombre_Output_entidades, Flag_csv, Ruta_Nombre_Output_entidades)


# # If flag_invente_dir3 == False
    else:

        Nombre_Output_entidades = Nombre_Input_entidades\
                                                    .union(Nombre_Intermedio_cruce1_entidades)
        
        # =============================================================================
        # Aplicación de metodo para almacenar el resultado (parquet, csv, o ambos inclusive)
        # =============================================================================
        save_csv_parquet(Nombre_Output_entidades, Flag_csv, Ruta_Nombre_Output_entidades)







