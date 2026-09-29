from pathlib import Path

import mock
import transformers as trf

from tuned_lens import TunedLens
from tuned_lens.scripts import ingredients as ing
from tuned_lens.scripts.train_loop import Train


def test_train_get_lens_records_model_revision(
    random_small_model: trf.PreTrainedModel, tmp_path: Path
):
    train = Train(
        model=ing.Model(name="test-model", revision="step512"),
        data=ing.Data(),
        opt=ing.Optimizer(),
        dist=ing.Distributed(),
        output=tmp_path,
    )

    with mock.patch.object(TunedLens, "from_model", wraps=TunedLens.from_model) as from_model:
        lens = train.get_lens(random_small_model)

    from_model.assert_called_once_with(
        random_small_model, model_revision="step512"
    )
    assert lens.config.base_model_revision == "step512"
