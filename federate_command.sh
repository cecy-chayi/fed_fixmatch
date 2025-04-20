# dynamic threshold on
python train.py --dataset fmnist --arch wideresnet --batch-size 32 --lr 0.03 --expand-labels --seed 5 --out results/fmnist@0.1_dynamic --total-cr 100 --num-clients 50 --frac 0.1 --local-ep 1 --eval-step 128 --dirichlet-alpha 0.8 --labeled-ratio 0.1 --wdecay 0.001 --use-dynamic-threshold --threshold 0.95 --final-threshold 0.85
# progressive
python train.py --dataset fmnist --arch wideresnet --batch-size 32 --lr 0.03 --expand-labels --seed 5 --out results/fmnist@0.1_progressive --total-cr 100 --num-clients 50 --frac 0.1 --local-ep 1 --eval-step 128 --dirichlet-alpha 0.8 --labeled-ratio 0.1 --wdecay 0.001 --use-progressive --pr 3
