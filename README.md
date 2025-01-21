# Privacy Source Code

To check out the source code and computation of t' check out "scripts/compute_t_dash.py"


Preperation: 
- Prepare dataset csv similar to ``example.csv''


### Run experiments 

Have a look at the batch scripts for the different experiments.
Below is a summary. 
The source code provided trains and samples a Latent Diffusion Model and performs computation of q in Latent space. 

---

### 1. Train Generative Model


```bash
python {predefined_path}/scripts/train_unconditional.py \
src/experiments/04_find_ring/find_ring.py \
find_ring
# Optional arguments (uncomment if needed):
# --data_csv=cxr14supportdevices.csv
# --data.limit_dataset_size=1778
# --data_csv=cxr14privacy.csv
```

### 2. Train af_classifier and id_classifier
```bash
python train_af_classifier.py src/experiments/04_find_ring/find_ring.py train_af_classifier_ring --use_synthetic_af --data_csv=celebahq_latent.csv
```


```bash
python train_id_classifier.py src/experiments/05_sunglasses/find_sunglasses.py train_id_classifier --data_csv=celebahq_latent.csv
```




### 3. Compute t'
```bash
python scripts/compute_t_dash.py \
    src/experiments/base_experiment.py \
    simple_model \
    --use_synthetic_af \
    --model_dir=final \
    --af_classifier_path=log/af_best.ckpt \
    --id_classifier_path=log/id_best.ckpt
```

