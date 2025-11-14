FROM python:3.11

WORKDIR /app/work

COPY . /app/work

RUN pip install --upgrade pip
RUN pip install .

CMD ["bash"]
