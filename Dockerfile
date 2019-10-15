
# Build with: docker build -t claudiofahey/tensorflow:19.03-py3-custom .

FROM nvcr.io/nvidia/tensorflow:19.03-py3

MAINTAINER Claudio Fahey <Claudio.Fahey@dell.com>

# Install SSH and various utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client \
        openssh-server \
        lsof \
    && \
    rm -rf /var/lib/apt/lists/*

# Configure SSHD for MPI.
RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
    echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/^Port 22/Port 2222/' /etc/ssh/sshd_config && \
    echo "HOST *" >> /root/.ssh/config && \
    echo "PORT 2222" >> /root/.ssh/config && \
    mkdir -p /root/.ssh && \
    ssh-keygen -t rsa -b 4096 -f /root/.ssh/id_rsa -N "" && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 700 /root/.ssh && \
    chmod 600 /root/.ssh/*

WORKDIR /scripts

EXPOSE 2222
