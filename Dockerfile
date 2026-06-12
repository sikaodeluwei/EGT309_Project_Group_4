FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Convert Windows CRLF line endings to Linux LF
RUN sed -i 's/\r$//' run.sh

CMD ["bash", "run.sh"]
