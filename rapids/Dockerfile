ARG FROM_IMAGE
FROM ${FROM_IMAGE}

MAINTAINER Claudio Fahey <Claudio.Fahey@dell.com>

ENV PATH=/opt/conda/envs/rapids/bin:$PATH

RUN conda install -p /opt/conda/envs/rapids -y \
        configargparse \
        python-snappy

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
