for var in {0..9}
do
  wget -nc https://storage.googleapis.com/lambdastruck_bucket/github_readmes/github_repos_00000000000$var.json.gz -O data/github_repos_00000000000$var.json.gz
done
