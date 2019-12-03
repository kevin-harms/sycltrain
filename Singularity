Bootstrap: shub
From: singularityhub/centos

%runscript
exec echo "dpc++ singularity tutorial image"
exec cat /etc/redhat-release

%environment

%setup

%post
# setup repo
tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel(R) oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
EOF
mv /tmp/oneAPI.repo /etc/yum.repos.d

yum update -y
yum install -y epel-release
yum install intel-basekit
yum install intel-hpckit

# setup demo code
mkdir -p /code
cd /code
git clone https://github.com/kevin-harms/sycltrain.git
