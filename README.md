# Show Me The Rhyme
## 2021-2 `자연어처리와 딥러닝 프로젝트`
### presentation link
- https://drive.google.com/file/d/19h2H4Pssenb4C0zb4dAOQpp36ELtF3uu/view?usp=sharing


----
- base line code 
- https://github.com/haven-jeon/KoBART-chatbot.git

### install 
```shell
python3 -m venv .venv
source .venv/bin.activate

pip install --upgrade pip
pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
pip install pandas
pip install pytorch_lightning==1.2.1
pip install sentence-transformers
```

### download pretrained model
```python
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
get_kobart_tokenizer(".")
get_pytorch_kobart_model(cachedir=".")
```

### train
```shell
source train_script.sh
```

### infer
```shell
source infer_script.sh
```
