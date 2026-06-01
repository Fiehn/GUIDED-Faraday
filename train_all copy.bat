


uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_deterministic.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_cnn_deterministic.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_deterministic.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_noUser.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guide_cnn.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guide.yaml"

uv run src/train.py "--config" "config/tests/agg_runs/diffusion.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/baseline_quantile.yaml"

ECHO "Returning to already trained models"

uv run src/train.py "--config" "config/tests/agg_runs/art.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guide_hierarchy.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guide_vamp.yaml"

uv run src/train.py "--config" "config/tests/agg_runs/cvae.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guide_hierarchy_cnn.yaml"

uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/faraday.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_relu.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_peak.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/guided_faraday_cnn.yaml"
uv run src/train.py "--config" "config/tests/agg_runs/baseline_mean.yaml"