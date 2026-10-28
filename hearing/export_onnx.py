"""Turns Meta's Omnilingual ASR CTC model into ONNX, once, so the studio hears on Windows with onnxruntime and needs
no Docker. Only whoever makes the model file runs this, in Docker, because fairseq2 runs on Linux only; the
checkpoint (omniASR-CTC-1B-v2.pt and omniASR_tokenizer_written_v2.model) in /models, the output folder in /out and
16 kHz mono wavs to check against in /check:

    docker build -t local-clip-studio-convert hearing
    docker run --rm -v <models>:/models:ro -v <out>:/out -v <check>:/check:ro local-clip-studio-convert

It writes model.onnx with its weights in model.onnx.data (float32), model.int8.onnx (its matrix products in 8 bits),
letters.json (what each token writes when Omnilingual's own decoder decodes it alone, as the hearing reads each
letter) and, for each wav in /check, <name>.npz: the samples the model was given and the logits it gave, which the
studio's check compares its own hearing with. The way through fairseq2 follows sherpa-onnx's exporter
(scripts/omnilingual-asr/export-onnx.py, Apache 2.0).
"""
import glob
import json
import os
import sys

import numpy as np
import onnx
import torch
from fairseq2.nn.batch_layout import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from onnxruntime.quantization import QuantType, quantize_dynamic

CARD = os.environ.get("LCS_HEARING_CARD", "omniASR_CTC_1B_v2")
OUT = "/out"


class Seen(ASRInferencePipeline):
    """The pipeline as the hearing ran it, keeping what the model was given and what it gave."""

    def _apply_model_wav2vec2asr(self, batch):
        layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens, device=batch.source_seqs.device)
        logits, _ = self.model(batch.source_seqs, layout)
        self.seen = (batch.source_seqs[0].float().numpy(), logits[0].float().numpy())
        return [""]


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x, BatchLayout(x.shape, seq_lens=[x.shape[1]], device=x.device))
        return logits


def main():
    torch.set_num_threads(8)
    pipe = Seen(model_card=CARD, device="cpu", dtype=torch.float32)
    with torch.inference_mode():
        for path in sorted(glob.glob("/check/*.wav")):
            with open(path, "rb") as f:
                pipe.transcribe([f.read()], batch_size=1)
            given, logits = pipe.seen
            np.savez_compressed(os.path.join(OUT, os.path.basename(path)[:-4] + ".npz"), given=given, logits=logits)
            print("checked", path, given.shape, logits.shape, flush=True)
        vocab = pipe.tokenizer.vocab_info.size
        letters = [pipe.token_decoder(torch.tensor([i])) for i in range(vocab)]
    with open(os.path.join(OUT, "letters.json"), "w", encoding="utf-8") as f:
        json.dump({"card": CARD, "letters": letters}, f, ensure_ascii=False)
    print("letters", vocab, flush=True)

    model = Logits(pipe.model).eval()
    os.makedirs("/tmp/first", exist_ok=True)
    first = "/tmp/first/model.onnx"
    torch.onnx.export(model, (torch.rand(1, 16000 * 10),), first, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n", 1: "samples"}, "logits": {0: "n", 1: "frames"}}, opset_version=17,
                      dynamo=False)
    onnx.save_model(onnx.load(first), os.path.join(OUT, "model.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.onnx.data")
    print("float32 written", flush=True)
    quantize_dynamic(os.path.join(OUT, "model.onnx"), os.path.join(OUT, "model.int8.onnx"),
                     op_types_to_quantize=["MatMul"], weight_type=QuantType.QUInt8)
    print("int8 written", flush=True)
    print("versions: torch", torch.__version__, "onnx", onnx.__version__, file=sys.stderr)


if __name__ == "__main__":
    main()
