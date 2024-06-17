Before running the main script, you need to generate the `.h5` file and the noise. To do this, run `tools.py` and `generate.py`:
```bash
python ./utils/tools.py
python ./noise/generate.py
```

Once the `.h5` file and noise are generated, you can run the main script `NRCH.py` to play with the model:
```bash
python NRCH.py
```

We have already provided the trained model under 50% noise in 64-bit on MIRFlickr-25K dataset.
