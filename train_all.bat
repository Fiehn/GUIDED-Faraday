uv run src/train.py "--config" "config/tests/phased/art.yaml"

uv run src/train.py "--config" "config/tests/phased/baseline_quantile.yaml"

uv run src/train.py "--config" "config/tests/phased/baseline_average.yaml"

uv run src/train.py "--config" "config/tests/phased/diffusion.yaml"

uv run src/train.py "--config" "config/tests/phased/phase_splitter.yaml"

uv run src/train.py "--config" "config/tests/phased/set_guide_vae.yaml"

uv run src/train.py "--config" "config/tests/phased/sequential.yaml"

shutdown /s /t 1