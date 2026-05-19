uv run src/train.py "--config" "config/tests/phased/set_guide_cnn.yaml"
uv run src/train.py "--config" "config/tests/phased/set_guide.yaml"
uv run src/train.py "--config" "config/tests/phased/art.yaml"
uv run src/train.py "--config" "config/tests/phased/diffusion.yaml"
uv run src/train.py "--config" "config/tests/phased/phase_splitter.yaml"
uv run src/train.py "--config" "config/tests/phased/sequential.yaml"
uv run src/train.py "--config" "config/tests/phased/baseline_average.yaml"
uv run src/train.py "--config" "config/tests/phased/baseline_quantile.yaml"

shutdown /s /t 1 