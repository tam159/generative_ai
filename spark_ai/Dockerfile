FROM langchain/langgraph-api:3.12

RUN apt update && apt upgrade -y
RUN apt install -y wget vim lsb-release gnupg2 procps
RUN wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt update && apt install -y temurin-8-jre
RUN wget https://dlcdn.apache.org/spark/spark-3.5.2/spark-3.5.2-bin-hadoop3-scala2.13.tgz
RUN tar -xvzf spark-3.5.2-bin-hadoop3-scala2.13.tgz -C /opt && rm spark-3.5.2-bin-hadoop3-scala2.13.tgz
ENV JAVA_HOME=/usr/lib/jvm/temurin-8-jre-arm64
ENV SPARK_HOME=/opt/spark-3.5.2-bin-hadoop3-scala2.13
ENV PATH=$PATH:$SPARK_HOME/bin

ADD . /deps/spark_ai

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"spark_lakehouse_agent": "/deps/spark_ai/spark_agent/lakehouse_agent.py:graph", "search_agent": "/deps/spark_ai/spark_agent/search_agent.py:graph"}'

WORKDIR /deps/spark_ai

CMD exec uvicorn langgraph_api.server:app --log-config /api/logging.json --host 0.0.0.0 --port $PORT --no-access-log --reload
