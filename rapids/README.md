# RAPIDS Benchmark Utilities

This provides benchmarks for ETL workloads on RAPIDS.
It is designed to run on a multiple DGX-2 servers with a NAS such as Isilon.

## Benchmark Procedure

### Disk Mounts

This document and associated scripts assume that each compute server mounts Isilon on
the mount points `/mnt/isilon`, `/mnt/isilon1`, `/mnt/isilon2`, ..., `/mnt/isilon16`.
The script [mount_isilon.py](../mount_isilon.py) can be used to perform the necessary mounts.

Each DGX-2 server should have a local disk mounted on `/raid`.

### Download this Repository

```
dgxuser@dgx2-1:~$
cd /mnt/isilon/data
git clone https://github.com/claudiofahey/ai-benchmark-util
cd ai-benchmark-util/rapids
```

### Download Mortgage Data

```
dgxuser@dgx2-1:~$

mkdir -p /mnt/isilon/data/mortgage
cd /mnt/isilon/data/mortgage
wget http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2016.tgz
tar -xzvf mortgage_2000-2016.tgz

ls -lhR
drwxr-xr-x 2 root    root    2.7K Nov 19 21:19 acq
-rw-r--r-- 1 root    root     24G Oct 20  2018 mortgage_2000-2016.tgz
-rw-r--r-- 1 dgxuser dgxuser 3.6K Oct 19  2018 names.csv
drwxr-xr-x 2 root    root    4.6K Nov 19 22:30 perf
./acq:
total 4.7G
-rw-rw-r-- 1 dgxuser dgxuser  28M Oct 19  2018 Acquisition_2000Q1.txt
...
./perf:
total 219G
-rw-r--r-- 1 dgxuser dgxuser  950M Oct  5  2018 Performance_2000Q1.txt
...
```

### Build ORC Dataset

This uses Apache Spark to build ORC datasets using a variety of parameters.
The output will be on the NAS.

```
dgxuser@dgx2-1:/mnt/isilon/data/ai-benchmark-util/rapids$
./start_spark_notebook.sh
```

Open your browser to Jupyter Notebook at http://localhost:8886.

Open the notebook [mortgage_etl_create_orc_spark_4.ipynb](mortgage_etl_create_orc_spark_4.ipynb).
Click Run -> Run All Cells.

Open the notebook [mortgage_etl_copy_files_1.ipynb](mortgage_etl_copy_files_1.ipynb).
Click Run -> Run All Cells.

### Copy dataset to local drives on DGX-2 servers

Create a file named `hosts` containing one host name (or IP address) per DGX-2 server. For example:

```
dgx2-1
dgx2-2
dgx2-3
```

```
./copy_data_to_local.sh
```

### Configure Benchmark

```
dgxuser@dgx2-1:/mnt/isilon/data/ai-benchmark-util/rapids$
sudo apt install python3-pip
pip3 install setuptools wheel
pip3 install --requirement requirements.txt
```

Edit the files:
- p3_test_driver.config.yaml
- testgen.py

### Build Docker Containers (Optional)

We use a customized container based on the nightly RAPIDS container.
It contains additional Python libraries and disables the automatic execution of Jupyter.
Skip this section if you do not need to further customize the image.

```
./build_docker_nightly.sh
```

### Run Benchmark using P3 Test Driver

```
dgxuser@dgx2-1:/mnt/isilon/data/ai-benchmark-util/rapids$
./testgen.py | p3_test_driver -t - -c p3_test_driver.config.yaml
```

### View Results

In Jupyter Notebook, open the notebook [rapids_results_analyzer_1.ipynb](rapids_results_analyzer_1.ipynb).
Click Run -> Run All Cells.

# See Also

- https://rapids.ai/
- https://docs.rapids.ai/datasets/mortgage-data

Copyright (c) Dell Inc., or its subsidiaries. All Rights Reserved.
