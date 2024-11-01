https://arrow.apache.org/docs/developers/cpp/building.html

* in cpp/build: (need to specify install folder at build preset creation time)

```
cmake .. --preset ninja-release-minimal -DARROW_PARQUET=ON -DARROW_FILESYSTEM=ON -DARROW_WITH_SNAPPY=ON -DARROW_DATASET=ON -DCMAKE_INSTALL_PREFIX=/export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed
```

* run the build as a job with the activated conda workspace (for the g++ version), need the extra memory (as compared to simply "cmake --build ." in the build directory)

```
sbatch -p dgxh100 -t 12:00:00 --mem-per-cpu 8G --output=output.log --wrap="cmake --build /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow/cpp/build"
```

* the install step is very quick:

```
cmake --install .
```

* Also need `libcrypto libssl libthrift libsnappy` (here specifically `libcrypto.so.3  libsnappy.so.1  libssl.so.3  libthrift.so.0.15.0`), best to obtain them from conda and put them in the same folder as the generated libarrow / libparquet.

* conda list:

```
# packages in environment at /export/home/acs/stud/v/vlad_adrian.ulmeanu/miniconda3/envs/cpp_libs:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
_sysroot_linux-64_curr_repodata_hack 3                   haa98f57_10  
arrow-cpp                 0.2.post                      0    conda-forge
aws-c-auth                0.6.19               h5eee18b_0    anaconda
aws-c-cal                 0.5.20               hdbd6064_0    anaconda
aws-c-common              0.8.5                h5eee18b_0    anaconda
aws-c-compression         0.2.16               h5eee18b_0    anaconda
aws-c-event-stream        0.2.15               h6a678d5_0    anaconda
aws-c-http                0.6.25               h5eee18b_0    anaconda
aws-c-io                  0.13.10              h5eee18b_0    anaconda
aws-c-mqtt                0.7.13               h5eee18b_0    anaconda
aws-c-s3                  0.1.51               hdbd6064_0    anaconda
aws-c-sdkutils            0.1.6                h5eee18b_0    anaconda
aws-checksums             0.1.13               h5eee18b_0    anaconda
aws-crt-cpp               0.18.16              h6a678d5_0    anaconda
aws-sdk-cpp               1.10.55              h721c034_0    anaconda
binutils_impl_linux-64    2.40                 h5293946_0  
boost-cpp                 1.85.0               h3c6214e_4    conda-forge
bzip2                     1.0.8                h5eee18b_6  
c-ares                    1.19.1               h5eee18b_0    anaconda
ca-certificates           2024.9.24            h06a4308_0  
gcc                       11.4.0              h602e360_13    conda-forge
gcc_impl_linux-64         11.4.0              h00c12a0_13    conda-forge
gflags                    2.2.2                h6a678d5_1    anaconda
glog                      0.5.0                h6a678d5_1    anaconda
gxx                       11.4.0              h602e360_13    conda-forge
gxx_impl_linux-64         11.4.0              h634f3ee_13    conda-forge
icu                       75.1                 he02047a_0    conda-forge
jemalloc                  5.2.1                h6a678d5_6  
kernel-headers_linux-64   3.10.0              h57e8cba_10  
krb5                      1.20.1               h143b758_1    anaconda
ld_impl_linux-64          2.40                 h12ee557_0  
libabseil                 20230802.1      cxx17_h59595ed_0    conda-forge
libboost                  1.85.0               h0ccab89_4    conda-forge
libboost-devel            1.85.0               h00ab1b0_4    conda-forge
libboost-headers          1.85.0               ha770c72_4    conda-forge
libbrotlicommon           1.0.9                h5eee18b_8    anaconda
libbrotlidec              1.0.9                h5eee18b_8    anaconda
libbrotlienc              1.0.9                h5eee18b_8    anaconda
libcurl                   7.88.1               hdc1c0ab_1    conda-forge
libedit                   3.1.20230828         h5eee18b_0    anaconda
libev                     4.33                 h7f8727e_1    anaconda
libevent                  2.1.12               hdbd6064_1  
libgcc                    14.2.0               h77fa898_1    conda-forge
libgcc-devel_linux-64     11.4.0             h8f596e0_113    conda-forge
libgcc-ng                 14.2.0               h69a702a_1    conda-forge
libgomp                   14.2.0               h77fa898_1    conda-forge
libgrpc                   1.58.1               h30d5116_0    conda-forge
libnghttp2                1.52.0               h61bc06f_0    conda-forge
libprotobuf               4.24.3               hf27288f_1    conda-forge
libsanitizer              11.4.0              h5763a12_13    conda-forge
libssh2                   1.10.0               hdbd6064_3  
libstdcxx                 14.2.0               hc0a3c3a_1    conda-forge
libstdcxx-devel_linux-64  11.4.0             h8f596e0_113    conda-forge
libstdcxx-ng              14.2.0               h4852527_1    conda-forge
libthrift                 0.21.0               h0e7cc3e_0    conda-forge
libzlib                   1.3.1                hb9d3cd8_2    conda-forge
lz4-c                     1.9.4                hcb278e6_0    conda-forge
ncurses                   6.4                  h6a678d5_0    anaconda
ninja                     1.12.1               h297d8ca_0    conda-forge
openssl                   3.3.2                hb9d3cd8_0    conda-forge
orc                       1.6.4                hc68dd99_0  
rapidjson                 1.1.0.post20240409      hac33072_1    conda-forge
re2                       2023.03.02           h8c504da_0    conda-forge
s2n                       1.3.27               hdbd6064_0    anaconda
snappy                    1.2.1                h6a678d5_0    anaconda
sysroot_linux-64          2.17                h57e8cba_10  
thrift-compiler           0.21.0               h5888daf_0    conda-forge
thrift-cpp                0.21.0               ha770c72_0    conda-forge
tzdata                    2024b                h04d1e81_0    anaconda
utf8proc                  2.6.1                h5eee18b_1    anaconda
xsimd                     13.0.0               h297d8ca_0    conda-forge
xz                        5.4.6                h5eee18b_1  
zlib                      1.3.1                hb9d3cd8_2    conda-forge
zstd                      1.5.6                ha6fb4c9_0    conda-forge
```

