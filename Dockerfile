FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy all files into the container
COPY --chown=user . $HOME/app

# Hugging Face exposes port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
