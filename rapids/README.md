# RAPIDS Benchmark Utilities

This provides benchmarks for ETL workloads on RAPIDS.
It is designed to run on a multiple DGX-2 servers with a NAS such as Isilon.

## Benchmark Procedure

### Download Mortgage Data

```
dgxuser@dgx2-1:~$
mkdir -p /mnt/isilon/data/mortgage
cd /mnt/isilon/data/mortgage

# Use below for the full benchmark with 17 years of data.
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
dgxuser@dgx2-1:~$
cd rapids
./start_spark_notebook.sh
```

Open your browser to Jupyter Notebook at http://localhost:8886.

Open the notebook [mortgage_etl_create_orc_spark_1.ipynb](mortgage_etl_create_orc_spark_1.ipynb).

Click Run -> Run All Cells.

### Copy datasets to local drives on DGX-2 servers

```
./copy_data_to_local.sh
```

### Configure Benchmark

```
dgxuser@dgx2-1:~$
cd rapids
sudo apt install python3-pip
pip3 install setuptools
pip3 install --requirement requirements.txt
```

Edit the files:
- build_docker_nightly.sh
- p3_test_driver.config.yaml
- testgen.py

### Build Docker Containers (Optional)

We use a customized container based on the nightly RAPIDS container from 2019-11-22.
It contains additional Python libraries and disables the automatic execution of Jupyter.
Skip this section if you do not need to further customize the image.

```
./build_docker_nightly.sh
```

### Run Benchmark using P3 Test Driver

```
dgxuser@dgx2-1:~$
cd rapids
./testgen.py | p3_test_driver -t - -c p3_test_driver.config.yaml
```

### View Results

In Jupyter Notebook, open the notebook [rapids_results_analyzer_1.ipynb](rapids_results_analyzer_1.ipynb).
Click Run -> Run All Cells.

# See Also

- https://rapids.ai/
- https://docs.rapids.ai/datasets/mortgage-data
