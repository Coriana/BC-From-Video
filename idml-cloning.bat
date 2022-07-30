@echo off
set DATA=Rl-From-Early-Game-2X-4.37.mp4
set MODEL=data/VPT-models/foundation-model-1x.model
set IN_WEIGHTS=data/VPT-models/foundation-model-1x.weights
set OUT_WEIGHTS=Corianas-rl-from-house
:: OpenAI VPT BC weight decay was = 0.039428
TITLE VPT IDML Backpropagating %DATA% into %IN_WEIGHTS%

python idml-cloning.py --video="%DATA%" --out-weights="train/%OUT_WEIGHTS%.weights" --in-model="%MODEL%" --in-weights="%IN_WEIGHTS%"


pause