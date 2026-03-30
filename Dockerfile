# Start from an official Python image — "slim" means smaller file size
FROM python:3.11-slim

# Set the working directory inside the container
# All future commands run from here
WORKDIR /app

# Copy requirements first — Docker caches layers
# If requirements don't change, this layer is reused (faster rebuilds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy your actual code
COPY . .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command to start your server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]