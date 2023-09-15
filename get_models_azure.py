# modify the following function to download all pickle files from azure blob storage and save in a models folder
# create a function to read pkl files from blob storage
import pickle
from azure.storage.blob import BlobServiceClient   
import os

def read_pkl_blob_azure(account_name, account_key, container_name):
    #create a client to interact with blob storage
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)    
    #use the client to connect to the container
    container_client = blob_service_client.get_container_client(container_name)
    # create the models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    # iterate over the blobs in the container
    for blob_i in container_client.list_blobs():
        # download the file pkl file and save in a models folder
        if blob_i.name.endswith('.pkl'):
            with open(os.path.join('models', blob_i.name), 'wb') as file:
                file.write(container_client.get_blob_client(blob_i.name).download_blob().readall())
            
            
# define the azure credentials
#enter credentials
account_name = os.environ.get('SECRET_NAME_BS_AZURE')  
account_key = os.environ.get('SECRET_KEY_BS_AZURE')  
              
read_pkl_blob_azure(account_name, account_key, 'models')
    