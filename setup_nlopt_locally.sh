NLOPTDIR="nlopt-2.6.1"
REPOS_DIR=$PWD

tar -zxvf nlopt.tar.gz
cd nlopt/$NLOPTDIR
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$REPOS_DIR/nlopt ..
make
make install
