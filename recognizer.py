"""
Stage 3 – Recognition
=====================
Two backends are supported:
  - ``crnn``  : CNN + BiLSTM + CTC  (fast, deployable on edge)
  - ``vlm``   : Vision-Language Model fine-tuned on ISO 6346 codes
                (higher accuracy, requires GPU with ≥8 GB VRAM)

CRNN Architecture
-----------------
  Input (1 × 64 × 512)
      │
  ResNet-34 backbone (feature extractor, last two stages)
      │  → (C × 1 × W')   feature map collapsed on height
      │
  BiLSTM × 2 layers  (hidden=256 each direction)
      │
  Linear projection → charset_size + 1  (blank token for CTC)
      │
  CTC beam-search decode → string
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CRNN model definition
# ──────────────────────────────────────────────────────────────────────

class _BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,
                           bidirectional=True, batch_first=True)
        self.fc  = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out)


class CRNN(nn.Module):
    """
    Lightweight CRNN for fixed-height (64 px) plate strips.

    Parameters
    ----------
    charset : str
        Characters the model can output.  A CTC blank token is added
        internally at index 0.
    hidden_size : int
        BiLSTM hidden units per direction.
    num_rnn_layers : int
        Number of stacked BiLSTM layers.
    """

    def __init__(
        self,
        charset: str,
        hidden_size: int = 256,
        num_rnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.charset  = charset
        self.num_classes = len(charset) + 1   # +1 for CTC blank

        # ── CNN backbone (ResNet-34 style, but height → 1 via pooling) ──
        self.cnn = self._build_cnn()

        # Feature map channels after CNN
        cnn_out_channels = 512

        # ── Sequence modelling ──
        self.rnn = nn.Sequential()
        in_size = cnn_out_channels
        for i in range(num_rnn_layers):
            out_size = hidden_size if i < num_rnn_layers - 1 else self.num_classes
            layer = _BidirectionalLSTM(in_size, hidden_size, out_size)
            self.rnn.add_module(f"lstm{i}", layer)
            in_size = out_size if i < num_rnn_layers - 1 else out_size
            if dropout > 0 and i < num_rnn_layers - 1:
                self.rnn.add_module(f"drop{i}", nn.Dropout(dropout))

        # Final projection to num_classes
        self.fc_out = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, H, W)

        Returns
        -------
        torch.Tensor  shape (T, B, num_classes)  – log-softmax over classes
        """
        feat = self.cnn(x)                         # (B, C, 1, W')
        feat = feat.squeeze(2)                     # (B, C, W')
        feat = feat.permute(2, 0, 1)               # (W', B, C)  = (T, B, C)

        # Run through stacked BiLSTMs
        out = feat.permute(1, 0, 2)                # (B, T, C)
        for name, module in self.rnn.named_children():
            out = module(out)

        out = self.fc_out(out)                     # (B, T, num_classes)
        out = out.permute(1, 0, 2)                 # (T, B, num_classes)
        return F.log_softmax(out, dim=2)

    @staticmethod
    def _build_cnn() -> nn.Sequential:
        """
        Simplified ResNet-34-style stack that maps
        (1, 64, W) → (512, 1, W//16).
        """
        def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        return nn.Sequential(
            conv_bn_relu(1,   64,  k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),                          # H/2
            conv_bn_relu(64,  128, k=3, s=1, p=1),
            nn.MaxPool2d(2, 2),                          # H/4
            conv_bn_relu(128, 256, k=3, s=1, p=1),
            conv_bn_relu(256, 256, k=3, s=1, p=1),
            nn.MaxPool2d((2, 1), (2, 1)),                # H/8
            conv_bn_relu(256, 512, k=3, s=1, p=1),
            nn.BatchNorm2d(512),
            conv_bn_relu(512, 512, k=3, s=1, p=1),
            nn.MaxPool2d((2, 1), (2, 1)),                # H/16 = 1 (for H=64)
            conv_bn_relu(512, 512, k=2, s=1, p=0),      # compress last spatial dim
        )


# ──────────────────────────────────────────────────────────────────────
# Recognition result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    text: str
    confidence: float       # mean character confidence [0, 1]
    raw_logits: Optional[np.ndarray] = None


# ──────────────────────────────────────────────────────────────────────
# Recogniser wrapper
# ──────────────────────────────────────────────────────────────────────

class ContainerRecogniser:
    """
    Thin wrapper that loads a CRNN or VLM backend and exposes a single
    ``recognise(plate_tensor)`` method.

    Parameters
    ----------
    cfg : dict
        ``recognition`` section of model_config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg     = cfg
        self.backend = cfg["backend"].lower()
        self.charset = cfg["crnn"]["charset"]
        self.device  = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.backend == "crnn":
            self._model = self._load_crnn(cfg)
        elif self.backend == "vlm":
            self._model = self._load_vlm(cfg)
        else:
            raise ValueError(f"Unknown recognition backend: {self.backend}")

        logger.info("Recogniser ready: %s  device=%s", self.backend, self.device)

    # ------------------------------------------------------------------

    def recognise(self, plate: np.ndarray) -> RecognitionResult:
        """
        Parameters
        ----------
        plate : np.ndarray
            Pre-processed plate tensor of shape (1, H, W) float32.

        Returns
        -------
        RecognitionResult
        """
        if self.backend == "crnn":
            return self._infer_crnn(plate)
        return self._infer_vlm(plate)

    # ------------------------------------------------------------------
    # CRNN inference
    # ------------------------------------------------------------------

    def _load_crnn(self, cfg: dict) -> CRNN:
        crnn_cfg = cfg["crnn"]
        model = CRNN(
            charset=crnn_cfg["charset"],
            hidden_size=crnn_cfg["rnn_hidden"],
            num_rnn_layers=crnn_cfg["rnn_layers"],
            dropout=crnn_cfg["dropout"],
        ).to(self.device)

        import os
        if os.path.exists(cfg["weights"]):
            state = torch.load(cfg["weights"], map_location=self.device)
            model.load_state_dict(state)
            logger.info("CRNN weights loaded from %s", cfg["weights"])
        else:
            logger.warning("CRNN weights not found – model is randomly initialised.")

        model.eval()
        return model

    def _infer_crnn(self, plate: np.ndarray) -> RecognitionResult:
        t = torch.from_numpy(plate).unsqueeze(0).to(self.device)  # (1,1,H,W)

        with torch.no_grad():
            log_probs = self._model(t)   # (T, 1, num_classes)

        probs = log_probs.exp().squeeze(1).cpu().numpy()   # (T, num_classes)
        text, conf = self._ctc_greedy_decode(probs)
        return RecognitionResult(text=text, confidence=conf, raw_logits=probs)

    def _ctc_greedy_decode(
        self, probs: np.ndarray
    ) -> Tuple[str, float]:
        """
        Greedy CTC decoder (blank = index 0).
        Returns the decoded string and the mean max-class probability.
        """
        blank = 0
        indices = np.argmax(probs, axis=1)   # (T,)
        confidences = probs[np.arange(len(indices)), indices]

        # Collapse repeats, then remove blanks
        chars, confs = [], []
        prev = None
        for idx, conf in zip(indices, confidences):
            if idx != prev:
                if idx != blank:
                    chars.append(self.charset[idx - 1])   # -1 because blank=0
                    confs.append(conf)
                prev = idx

        text = "".join(chars)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return text, mean_conf

    # ------------------------------------------------------------------
    # VLM inference
    # ------------------------------------------------------------------

    def _load_vlm(self, cfg: dict):
        """Lazy-load a VLM (Qwen2-VL or similar) via transformers."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
        except ImportError as exc:
            raise ImportError("pip install transformers accelerate") from exc

        vlm_cfg = cfg["vlm"]
        processor = AutoProcessor.from_pretrained(vlm_cfg["model_id"])
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_cfg["model_id"],
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return {"model": model, "processor": processor, "cfg": vlm_cfg}

    def _infer_vlm(self, plate: np.ndarray) -> RecognitionResult:
        from PIL import Image  # type: ignore

        # Reconstruct uint8 image from normalised tensor
        img_np = ((plate[0] * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        proc  = self._model["processor"]
        model = self._model["model"]
        vlm_cfg = self._model["cfg"]

        inputs = proc(
            text=vlm_cfg["prompt"],
            images=pil_img,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=vlm_cfg["max_new_tokens"],
            )

        text = proc.decode(out[0], skip_special_tokens=True).strip().upper()
        # VLMs don't give token-level probs easily; use a fixed proxy
        return RecognitionResult(text=text, confidence=0.95)
