# Use Python 3.9 image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Create a writable directory for non-root user (Hugging Face requirement)
RUN mkdir -p /code/cache && chmod -R 777 /code/cache

# Set environment variable for cache
ENV TRANSFORMERS_CACHE=/code/cache

# ... (rest of your Dockerfile)
ENV PYTHONPATH=/code
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]

# Command to run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]