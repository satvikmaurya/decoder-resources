cd /qec/liblsqecc/external/rotation-decomposer/newsynth/
make buildhs
cd -
cd /qec/liblsqecc/
mkdir -p build
cd build
cmake .. -DUSE_GRIDSYNTH=1
make -j2
