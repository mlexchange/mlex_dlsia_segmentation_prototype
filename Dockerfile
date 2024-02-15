FROM python:3.9

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app/work/
COPY src/ src/
CMD ["bash"]