# Develop SentencePiece

## Install dependencies

Clone SentencePiece repo:

```sh
git clone https://github.com/google/sentencepiece.git
```

Install dependencies:

```sh
sudo apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev protobuf-compiler libprotobuf-dev
```

(Optional) Update protobuf to 3.14.0 or later:

```sh
mkdir -p ~/repos/sentencepiece/bin
cd ~/repos/sentencepiece/bin
curl --proxy http://127.0.0.1:11111 -sLo protoc-3.14.0-linux-x86_64.zip https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-linux-x86_64.zip
unzip -j protoc-3.14.0-linux-x86_64.zip 'bin/protoc'
```

Check `protoc` version:

```sh
cd ~/repos/sentencepiece
./bin/protoc --version
```

* protobuf upgrade to 3.14.0 · Issue #595 · google/sentencepiece
  * https://github.com/google/sentencepiece/issues/595

## Build Python bindings from C++ source

* Build and install SentencePiece command line tools from C++ source
  * https://github.com/google/sentencepiece?tab=readme-ov-file#build-and-install-sentencepiece-command-line-tools-from-c-source

* SentencePiece Python Wrapper
  * https://github.com/google/sentencepiece/blob/master/python/README.md

(Optional) Re-generate proto (`.pb.h`, `.pb.cc`) if `.proto` is modified

```sh
cd ~/repos/sentencepiece/src
../bin/protoc --cpp_out=builtin_pb/ sentencepiece_model.proto
../bin/protoc --python_out=../python/src/sentencepiece sentencepiece_model.proto
```

Cmake build and install:

```sh
cd sentencepiece
mkdir build && cd build
cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=./root
# make -j $(nproc)
# sudo make install
# sudo ldconfig -v
make install
```

Build python bindings:

```sh
cd ../python && python setup.py build && python setup.py install
```

Install via wheel:

```sh
# pip install dist/sentencepiece*.whl
```

Or install as editable package:

```sh
pip install -e .
```

## Check built files

```sh
ls -hal ~/repos/sentencepiece/python/src/sentencepiece
ls -hal ../build/root
```

The following files should be modified latest:

```sh
_sentencepiece.cpython-311-x86_64-linux-gnu.so
```

## Uninstall
  
Suitable for both editable and wheel installs:

```sh
pip uninstall sentencepiece
```

and reinstall is same as above.

## Making changes to SentencePiece source codes

...

## Re-build when source code changes

```sh
cd ../build
cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=./root
make install
cd ../python
rm -rf build/ dist/
python setup.py build && pip install -e .
```

Dump to `rebuid.sh` for quick run.
See: [sentencepiece/python/rebuild.sh](https://github.com/Hansimov/sentencepiece/blob/master/python/rebuild.sh)