mkdir train
cd train
mkdir ok
mkdir nok
cd ..
mkdir test
cd test
mkdir ok
mkdir nok
cd ..
mkdir valid
cd valid
mkdir ok
mkdir nok
cd ..
cd train/
cd ok
mv ../../ok/train/* .
cd ..
cd nok
mv ../../nok/train/* .
cd ../..
cd valid
cd ok
mv ../../ok/valid/* .
cd ..
cd nok
mv ../../nok/valid/* .
cd ../..
cd test/ok/
mv ../../ok/test/* .
cd ..
cd nok
mv ../../nok/test/* .
cd ../..
