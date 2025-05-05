FROM python:3.11.8-slim
ADD ./requirements.txt .
RUN python -m pip install -r requirements.txt
COPY ./data ./data
COPY ./models ./models
COPY ./visualizations ./visualizations
COPY ./run_training.py .
COPY ./run_testing.py .
COPY ./config.py .
