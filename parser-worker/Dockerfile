# STAGE 1: Build the Parser JAR
FROM maven:3.9.6-eclipse-temurin-21-alpine AS build
WORKDIR /build

RUN apk add --no-cache git
RUN git clone https://github.com/odota/parser.git .

# Upgrade maven-shade-plugin from 2.2 → 3.5.1 to fix corrupt JAR bug
# (2.2 fails with modern protobuf/service-loader files from clarity deps)
RUN sed -i 's/<version>2.2<\/version>/<version>3.5.1<\/version>/' pom.xml

RUN mvn clean package -DskipTests

# STAGE 2: Final Worker Image  
FROM eclipse-temurin:21-jre

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/target/stats-0.1.0.jar /app/parser.jar

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

COPY worker.py .
COPY app.py .

ENV PORT=8080
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]