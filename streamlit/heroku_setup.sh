mkdir -p .streamlit
cat streamlit/config.toml | sed s/PORT/$PORT/ - > .streamlit/config.toml
cp streamlit/credentials.toml .streamlit/credentials.toml
pip install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
nbdev_build_lib; pip install -e .
