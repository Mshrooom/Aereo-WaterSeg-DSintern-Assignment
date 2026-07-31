# SegFormer deployment patch

Copy these files into the root of the local `aereo-water-segmentation` repository.
The patch adds a CPU Docker API for the trained Experiment C checkpoint.

Required local model path (ignored by Git):

```text
artifacts/checkpoints/segformer_best/
  config.json
  metadata.pt
  model.safetensors OR pytorch_model.bin
```

Build and start:

```powershell
docker compose -f docker-compose.segformer.yml up --build -d
```

Test:

```powershell
.\scripts\test_segformer_api.ps1 -ImagePath "D:\path\sample.png"
```

Stop:

```powershell
docker compose -f docker-compose.segformer.yml down
```
