import ast
from pathlib import Path

import numpy as np

from agent_act_reproduction.agent import MockAgent
from agent_act_reproduction.config import ACTION_FEATURES, ENV_STATE_FEATURES, ROBOT_STATE_FEATURES
from agent_act_reproduction.data.dataset import BottleACTDataset, BottleDatasetArrays
from agent_act_reproduction.runtime import PolicyRouter


def test_mock_agent_returns_structured_supported_skill() -> None:
    response = MockAgent().select("把瓶子放到架子上")
    assert response.status == "OK"
    assert response.request is not None
    assert response.request.to_dict() == {"skill": "place_bottle_on_shelf", "args": {}}


def test_mock_agent_refuses_unknown_capability() -> None:
    response = MockAgent().select("Draw a circle.")
    assert response.status == "CANNOT_EXECUTE"
    assert response.request is None
    assert response.available_capabilities == ("place_bottle_on_shelf",)


def test_schemas_have_fixed_dimensions_and_unique_names() -> None:
    assert len(ROBOT_STATE_FEATURES) == 4
    assert len(ENV_STATE_FEATURES) == 11
    assert len(ACTION_FEATURES) == 4
    assert len(set((*ROBOT_STATE_FEATURES, *ENV_STATE_FEATURES, *ACTION_FEATURES))) == 19


def test_policy_router_is_one_to_one(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bottle.pt"
    checkpoint.touch()
    router = PolicyRouter({"place_bottle_on_shelf": checkpoint})
    assert router.checkpoint_for("place_bottle_on_shelf") == checkpoint.resolve()


def test_action_chunks_pad_at_episode_boundary() -> None:
    arrays = BottleDatasetArrays(
        observation_state=np.zeros((2, 4), dtype=np.float32),
        observation_environment_state=np.zeros((2, 11), dtype=np.float32),
        action=np.asarray([[1, 2, 3, -1], [4, 5, 6, 1]], dtype=np.float32),
        episode_id=np.zeros(2, dtype=np.int32),
        episode_lengths=np.asarray([2], dtype=np.int32),
        metadata=[],
        feature_names={},
    )
    normalization = {
        "observation.state": {"mean": np.zeros(4), "std": np.ones(4)},
        "observation.environment_state": {"mean": np.zeros(11), "std": np.ones(11)},
        "action": {"mean": np.zeros(4), "std": np.ones(4)},
    }
    dataset = BottleACTDataset(
        arrays,
        episode_ids={0},
        chunk_size=4,
        normalization=normalization,
    )
    final_item = dataset[1]
    np.testing.assert_allclose(
        final_item["action"].numpy(),
        np.tile(arrays.action[1], (4, 1)),
    )
    assert final_item["action_is_pad"].tolist() == [False, True, True, True]


def test_runtime_has_no_expert_import() -> None:
    runtime_path = (
        Path(__file__).parents[1]
        / "agent_act_reproduction"
        / "runtime"
        / "execute.py"
    )
    tree = ast.parse(runtime_path.read_text())
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert all("expert" not in module for module in imported_modules)
