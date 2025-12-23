# Example SD Med 3.5 using Cog

1. The weights were downloaded locally using `hf`
2. Able to run on RTX 5070TI 16GB
3. Able to run on Replicate T4 16GB and L40S 48GB

## Running

Once weights are downloaded using `hf` you can run:

1. `cog predict -i prompt="a man in a futuristic space suit and a dragonfly head playing a DJ set at a huge concert"` 
2. `cog serve` 
3. `cog build --separate-weights --tag <your-tag>` 
4. `cog push` 