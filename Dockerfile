FROM python:3.11
RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install numpy
RUN pip install scipy 
RUN pip install sqlalchemy psycopg2
RUN pip install scikit-learn
RUN pip install requests
RUN pip install pyarrow
RUN pip install fastparquet
WORKDIR /app
copy src/ src/
copy data/ data/
ENTRYPOINT python src/functions.py && python src/main.py
