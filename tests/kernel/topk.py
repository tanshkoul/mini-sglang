from minisgl.kernel import fast_topk
from minisgl.utils import call_if_main


@call_if_main(__name__)
def test_fast_topk():
    import torch

    torch.manual_seed(0)
    B = 10
    clip = 50000
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(B, 100000, dtype=torch.float32, device="cuda").abs()
    indices = torch.full((B, 2048), -2, dtype=torch.int32, device="cuda")
    lengths = torch.full((B,), clip, dtype=torch.int32, device="cuda")
    fast_topk(score, indices, lengths)
    # sort indices by last dimension
    indices = indices.sort(dim=-1).values
    # find the pos where -2 is in indices
    answer = torch.topk(score[:, :clip], 2048, dim=-1, sorted=False).indices.sort(dim=-1).values

    # check how many different in each row
    indice_cpu = indices.cpu().tolist()
    answer_cpu = answer.cpu().tolist()

    for i in range(B):
        diff = set(indice_cpu[i]) - set(answer_cpu[i])
        if len(diff) > 0:
            print(f"row {i} has {len(diff)} different: {diff}")

    # test performance
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    # use a large GEMM to warm up GPU
    def perf(f):
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        _ = a @ b
        tic.record()
        for _ in range(100):
            f()
        toc.record()
        torch.cuda.synchronize()
        return tic.elapsed_time(toc) / 100

    t0 = perf(lambda: fast_topk(score, indices, lengths))
    t1 = perf(lambda: torch.topk(score[:, :clip], 2048, dim=-1, sorted=False))
    print(f"fast_topk: {t0} ms, torch.topk: {t1} ms")
