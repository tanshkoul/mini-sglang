from minisgl.kernel import fast_topk
from minisgl.utils import call_if_main


@call_if_main(__name__)
def test_fast_topk():
    import torch

    score = torch.randn(10, 10000, dtype=torch.float32, device="cuda")
    indices = torch.empty(10, 2048, dtype=torch.int32, device="cuda")
    lengths = torch.full((10,), 10, dtype=torch.int32, device="cuda")
    fast_topk(score, indices, lengths)
    print(f"{indices=}")
