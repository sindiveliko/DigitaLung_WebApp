FROM python:3.10.6
EXPOSE 8501

WORKDIR /apppython

RUN /usr/local/bin/python -m pip install --upgrade pip

COPY digitalung digitalung
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


CMD [ "/bin/sh" , "-c" , "cd digitalung && streamlit run digitalung.py"]
