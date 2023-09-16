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

# copy the api code
COPY api/ ./api

# copy the models trained
COPY models/ ./models

# copy initializer file
COPY initializer.sh .

RUN chmod +x initializer.sh

# define the point of entry
EXPOSE 8000
ENTRYPOINT ["./initializer.sh"]
