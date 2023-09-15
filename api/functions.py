#import needed libraries
import pickle
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import urllib

# create a function to read pkl files from blob storage
def read_pkl_blob_azure(account_name, account_key, container_name, file_name):
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
    with urllib.request.urlopen(sas_url) as url:
        data = pickle.loads(url.read())
    return data
