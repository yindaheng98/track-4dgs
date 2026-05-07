# track-4dgs

```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master --no-build-isolation
pip install --upgrade git+https://github.com/facebookresearch/vggt.git@main
pip install --upgrade Pillow hydra-core omegaconf # deps for vggt
pip install --upgrade git+https://github.com/jytime/LightGlue.git#egg=lightglue # deps for vggt
```

## Download pth

cotracker
```shell
wget -P checkpoints https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

vggt
```sh
wget -P checkpoints/ https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/vggt_1B_commercial.pt --header="Authorization: Bearer $HF_TOKEN"
```
