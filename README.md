# BIC project
## Setup
```
pip install -r requirements.txt
```
## Reproducing experiments

Evaluating architectures - training from scratch
```
python train.py --model mobilenet --fixed_logo --epochs 5 --batch_size 128 --lr 0.001 --train_steps 300 --val_steps 100 --epochs 20
python train.py --model resnet18 --fixed_logo --epochs 5 --batch_size 128 --lr 0.001 --train_steps 300 --val_steps 100 --epochs 20
python train.py --model vit --fixed_logo --epochs 5 --batch_size 128 --lr 0.001 --train_steps 300 --val_steps 100 --epochs 20
```

Evaluating architectures - finetuning pretrained checkpoint
```
python train.py --model mobilenet --pretrained --fixed_logo --epochs 5 --batch_size 128 --lr 0.00001 --train_steps 300 --val_steps 100 --epochs 20
python train.py --model resnet18 --pretrained --fixed_logo --epochs 5 --batch_size 128 --lr 0.00001 --train_steps 300 --val_steps 100 --epochs 20
python train.py --model vit --pretrained --fixed_logo --epochs 5 --batch_size 128 --lr 0.00001 --train_steps 300 --val_steps 100 --epochs 20
```

Evaluating the impact of changing logo size

```
python train.py --model mobilenet --epochs 5 --batch_size 128 --lr 0.001 --train_steps 300 --val_steps 100 --epochs 20
python train.py --model resnet18 --epochs 5 --batch_size 128 --lr 0.001 --train_steps 300 --val_steps 100 --epochs 20
python train.py --model vit --epochs 5 --batch_size 128 --lr 0.001 --train_steps 300 --val_steps 100 --epochs 20
```