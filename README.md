# My-EACL
- More details can be found in paper : Fangxu Yu, Junjie Guo, Zhen Wu, Xinyu Dai, "[Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation](https://arxiv.org/abs/2403.20289)" 

## Code
**1) Download this GitHub**
```
git clone https://github.com/Yu-Fangxu/EACL.git
```

**2) Setup Environment**

We recommend creating a new environment:
```bash
conda create -n EACL python==3.9
conda activate EACL
```

Then install all the dependencies:
```
pip install -r requirements.txt
```

**3) Run Command for EACL**

```
bash run.sh IEMOCAP|MELD|EmoryNLP 'princeton-nlp/sup-simcse-roberta-large'|'YuxinJiang/sup-promcse-roberta-large'|'microsoft/deberta-large'

bash run.sh IEMOCAP 'princeton-nlp/sup-simcse-roberta-large'
bash run.sh IEMOCAP 'princeton-nlp/sup-simcse-roberta-base'

bash run.sh MELD 'YuxinJiang/sup-promcse-roberta-large'
bash run.sh MELD 'YuxinJiang/sup-promcse-roberta-base'

bash run.sh EmoryNLP 'microsoft/deberta-large'
bash run.sh EmoryNLP 'microsoft/deberta-base'
```

You could choose one dataset from IEMOCAP | MELD | EmoryNLP, and choose one base model from SimCSE | PromCSE | Deberta

<br> **If you find our repository helpful to your research, please consider citing:** <br>
```
@article{yu2024emotion,
  title={Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation},
  author={Yu, Fangxu and Guo, Junjie and Wu, Zhen and Dai, Xinyu},
  journal={arXiv preprint arXiv:2403.20289},
  year={2024}
}
```