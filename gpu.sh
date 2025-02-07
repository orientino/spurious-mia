savedir="exp"

for i in {0..16}
do
    python3 train_spurious.py --epochs 100 --lr 0.001 --wd 0.01 --n_shadows 16 --shadow_id $i --model resnet50 --data waterbirds --savedir $savedir
done

python3 inference.py --data waterbirds --savedir $savedir
python3 score.py --data waterbirds --savedir $savedir
python3 stats.py --data waterbirds 
