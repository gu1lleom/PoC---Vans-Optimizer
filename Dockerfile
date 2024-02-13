# Usar una imagen base oficial de Python
FROM python:3.9

# Establecer el directorio de trabajo en el contenedor
WORKDIR /code

# Copiar los archivos del proyecto al contenedor
COPY ./code /code

RUN pip install -r requirements.txt