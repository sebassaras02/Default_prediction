# load python image
FROM python:3.8.5

# define the working directory
WORKDIR /app

# copy the requirements file
COPY requirements.txt .

# update pip version
RUN pip install --upgrade pip

# install the requirements
RUN pip install -r requirements.txt

# copy get_models_azure.py to download the pkl models from azure
COPY get_models_azure.py .
# run the python script to download the models
RUN python get_models_azure.py

# copy the api code
COPY api/ ./api

# copy initializer file
COPY initializer.sh .

# define the point of entry
EXPOSE 8000
ENTRYPOINT ["./initializer.sh"]
