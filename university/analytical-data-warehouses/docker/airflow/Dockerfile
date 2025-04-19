FROM apache/airflow:2.5.1
USER root
RUN apt-get update \
    && apt-get install -yqq --no-install-recommends \
        freetds-bin \
        krb5-user \
        ldap-utils \
        libsasl2-2 \
        libsasl2-modules \
        libssl1.1 \
        lsb-release \
        sasl2-bin \
        sqlite3 \
        unixodbc \
        apt-utils \
        curl \
        libpq5 \
        freetds-bin \
        rsync \
        vim \
        netcat \
        locales \
        default-libmysqlclient-dev \
    && sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow
RUN python -m pip install --upgrade pip
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
WORKDIR /opt/airflow/dags