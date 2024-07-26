# Utiliser UBI 8 comme image de base
FROM redhat/ubi8:8.10

# Définir le répertoire de travail
WORKDIR /app

RUN yum install -y git git-lfs python39 python39-pip \
    && git lfs install \
    && git clone https://github.com/PaulMouly/Projet7withCSV.git \
    && cd ./HerokuApiDevelopment \
    && git lfs pull -I ./app/data/X_predictionV1.csv \
    && cd ./app \
    && pip3 install --no-cache-dir --upgrade -r ./requirements.txt 

ENTRYPOINT [ "python3" ]

CMD ["/app/HerokuApiDevelopment/app/app.py"]