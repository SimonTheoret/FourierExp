#! /usr/bin/env nix-shell
#! nix-shell -i bash -p bash

# TODO: make adv dataset runnable!!!

# AllCNN experiments

# adamw
# python3 -m main --exp_name=allcnnvanillaadamw --model_name=allcnn --optim_name=adamw --dataset_name=vanilla
# python3 -m main --exp_name=allcnngaussianadamw --model_name=allcnn --optim_name=adamw --dataset_name=gaussian
# python3 -m main --exp_name=allcnnadvadamw --model_name=allcnn --optim_name=adamw --dataset_name=adv --adv=true
# #sgd
# python3 -m main --exp_name=allcnnvanillasgd --model_name=allcnn --optim_name=sgd --dataset_name=vanilla
# python3 -m main --exp_name=allcnngaussiansgd --model_name=allcnn --optim_name=sgd --dataset_name=gaussian
python3 -m main --exp_name=allcnnadvsgd --model_name=allcnn --optim_name=sgd --dataset_name=adv --adv=true --seeds_range=3 #worst model on low pass

#MobileViT experiments

# adamw
# python3 -m main --exp_name=mobilevitadamwvanill --model_name=mobilevit --optim_name=adamw --dataset_name=vanilla
# python3 -m main --exp_name=mobilevitadamwgaussian --model_name=mobilevit --optim_name=adamw --dataset_name=gaussian
# python3 -m main --exp_name=mobilevitadamwadv --model_name=mobilevit --optim_name=adamw --dataset_name=adv --adv=true #too costly
#sgd
# python3 -m main --exp_name=mobilevitsgdvanilla --model_name=mobilevit --optim_name=sgd --dataset_name=vanilla
# python3 -m main --exp_name=mobilevitsgdgaussian --model_name=mobilevit --optim_name=sgd --dataset_name=gaussian
# python3 -m main --exp_name=mobilevitsgdadv --model_name=mobilevit --optim_name=sgd --dataset_name=adv --adv=true #too costly
