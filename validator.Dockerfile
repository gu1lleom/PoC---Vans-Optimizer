# Usar una imagen base oficial de Python
FROM python:3.9

# Instalar flake8 y MyPy
RUN pip install flake8==7.0.0 mypy==1.8.0

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Opcional: Si tu proyecto requiere dependencias adicionales, instálalas aquí
# RUN pip install -r requirements.txt

# Ejecutar flake8 para análisis estático del código
RUN flake8 --exclude=venv*,__pycache__ --max-line-length=120 . || echo "Failed flake8"

# Ejecutar MyPy para la verificación de tipos
RUN mypy .
