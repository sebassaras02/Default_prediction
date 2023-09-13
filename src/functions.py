import numpy as np
import matplotlib.pyplot as plt
import phik
import seaborn as sns
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import pandas as pd
import pickle

# create a function to make phik matrix
def phik_matriz(df : pd.DataFrame) -> list:
    df = df.drop(['ID'], axis = 1)
    matrix = df.phik_matrix()
    val = matrix.query('Status > 0.1').index.tolist()
    return val

# create a function to read data from azure bucket
def read_data(account_name, account_key, container_name, file_name):
    #create a client to interact with blob storage
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # select the file to read
    file_name = file_name
    # generate a shared access signature for the selected blob file    
    sas_i = generate_blob_sas(account_name = account_name,
                            container_name = container_name,
                            blob_name = file_name,
                            account_key=account_key,
                            permission=BlobSasPermissions(read=True),
                            expiry=datetime.utcnow() + timedelta(hours=1))
    #create the url to read the file
    sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + file_name + '?' + sas_i
    # read the file
    df = pd.read_csv(sas_url)
    return df


# create a function to save files into azure bucker
def save_file_blob_azure(account_name, account_key, object_to_save, name_object):
    # Conecta con la cuenta de Azure Blob Storage
    blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)
    container_name = 'models'
    # Accede o crea un contenedor si no existe
    container_client = blob_service_client.get_container_client(container_name)
    # Serializar el objeto a pickle
    contenido_pickle = pickle.dumps(object_to_save)
    blob_name = name_object
    # Sube el objeto pickle al contenedor
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(contenido_pickle, overwrite=True)
