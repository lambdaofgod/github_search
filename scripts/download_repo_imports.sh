for var in {0..9}
do
  wget -nc https://storage.googleapis.com/lambdaofgod_datasets/github/python_imports00000000000$var.csv.gz -O data/python_imports00000000000$var.csv.gz
done
for var in {10..99}
do
  wget -nc https://storage.googleapis.com/lambdaofgod_datasets/github/python_imports0000000000$var.csv.gz -O data/python_imports0000000000$var.csv.gz
done
