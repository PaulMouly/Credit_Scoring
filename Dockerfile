# Utiliser UBI 8 comme image de base
FROM redhat/ubi8:8.10

# Définir le répertoire de travail
WORKDIR /app

RUN yum install -y git git-lfs python39 python39-pip \
    && git lfs install \
    && git clone -b api-development https://github.com/PaulMouly/Projet7withCSV.git \
    && cd ./Projet7withCSV \
    && git lfs pull -I ./API/data/X_predictionV1.csv \
    && cd ./API \
    && pip3 install --no-cache-dir --upgrade -r ./requirements.txt 

ENTRYPOINT [ "python3" ]

CMD ["/app/Projet7withCSV/API/app.py"]