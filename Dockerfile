### Get Linux
FROM alpine:3.7

### Get Java via the package manager
RUN apk update \
&& apk upgrade \
&& apk add --no-cache bash \
&& apk add --no-cache --virtual=build-dependencies unzip \
&& apk add --no-cache curl \
&& apk add --no-cache openjdk8-jre

### Get Python, PIP

RUN apk add --no-cache python3 \
&& python3 -m ensurepip \
&& pip3 install --upgrade pip setuptools \
&& rm -r /usr/lib/python*/ensurepip && \
if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
if [[ ! -e /usr/bin/python ]]; then ln -sf /usr/bin/python3 ; fi && \
rm -r /root/.cache

### Get Flask for the app
#RUN pip install
#--trusted-host pypi.python.org flask
RUN pip3 install --upgrade pip
RUN apk add --update  python python3 python-dev python3-dev gfortran py-pip build-base
RUN apk update && apk add --no-cache libc6-compat
RUN BLAS=~/src/BLAS/libfblas.a LAPACK=~/src/lapack-3.5.0/liblapack.a pip install -v numpy==1.14

RUN  pip3 install wheel
RUN  pip3 install pyspark==2.3.2 --no-cache-dir
RUN  pip3 install findspark
RUN  pip3 install numpy

COPY docker_testing.py docker_testing.py
COPY TestDataset.csv TestDataset.csv
COPY TrainingDataset.csv TrainingDataset.csv
COPY wine_train_model.model wine_train_model.model
RUN ls -la /*
#### OPTIONAL :  SET JAVA_HOME environment variable, uncomment the line below if you need it
#ENV JAVA_HOME="/usr/lib/jvm/java-1.8-openjdk"

####

#ADD test.py /
CMD ["python3", "docker_testing.py"]