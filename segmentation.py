"""
segmentation.py

Loads a pretrained U-Net model (.h5 or torchscript/.pt/.pth) and runs inference on frames.

Exposes `SegmentationModel` class with:
 - `load()` to load the model file
 - `predict_mask(frame)` to return a binary mask (numpy uint8, same HxW as input)

Notes:
 - For Keras (.h5) the file is loaded with `tensorflow.keras.models.load_model`.
 - For PyTorch:
    - If you provide a TorchScript file (.pt, .torchscript) we load it with `torch.jit.load`.
    - If you have a .pth state_dict, pass a `model_builder` callable that returns the model instance
      so we can `load_state_dict`.
 - No code runs on import beyond definitions.
"""

from typing import Optional, Tuple, Callable
import os
import cv2
import numpy as np


class SegmentationModel:
    """Load and run a segmentation (U-Net) model saved as Keras(.h5) or PyTorch(.pt/.pth).

    Example:
        # Keras
        m = SegmentationModel('unet_best.h5', input_size=(256,256))
        m.load()
        mask = m.predict_mask(frame)

        # PyTorch script
        m = SegmentationModel('unet_script.pt', framework='torch', input_size=(256,256))
        m.load()
        mask = m.predict_mask(frame)

        # PyTorch state_dict with model builder
        m = SegmentationModel('model.pth', framework='torch', model_builder=my_builder, input_size=(256,256))
        m.load()
        mask = m.predict_mask(frame)
    """

    def __init__(self,
                 model_path: str,
                 framework: str = 'auto',
                 input_size: Tuple[int, int] = (256, 256),
                 model_builder: Optional[Callable[[], object]] = None,
                 device: str = 'cpu'):
        """
        Args:
            model_path: path to model file (.h5, .pt, .pth, .torchscript)
            framework: 'auto'|'keras'|'torch'
            input_size: (height, width) the model expects for inference
            model_builder: callable that returns a torch.nn.Module (required for loading .pth state_dict)
            device: 'cpu' or 'cuda'
        """
        self.model_path = model_path
        self.framework = framework
        self.input_size = tuple(input_size)
        self.model_builder = model_builder
        self.device = device

        # Loaded model handles
        self._keras_model = None
        self._torch_model = None
        self._torchscript = None
        self._torch = None

    def _infer_framework_from_ext(self) -> str:
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext in ('.h5', '.keras'):
            return 'keras'
        if ext in ('.pt', '.pth', '.torchscript'):
            return 'torch'
        return 'keras'

    def load(self):
        """Load model from disk. Call before `predict_mask`."""
        fw = self.framework
        if fw == 'auto':
            fw = self._infer_framework_from_ext()

        if fw == 'keras':
            # Lazy import to avoid requiring TF if user only uses PyTorch
            try:
                from tensorflow.keras.models import load_model
            except Exception as e:
                raise RuntimeError('TensorFlow/Keras is required to load .h5 models') from e
            self._keras_model = load_model(self.model_path)
            return

        if fw == 'torch':
            try:
                import torch
                self._torch = torch
            except Exception as e:
                raise RuntimeError('PyTorch is required to load torch models') from e

            # Try to load as torchscript first
            try:
                try:
                    self._torchscript = torch.jit.load(self.model_path, map_location=self.device)
                    self._torchscript.eval()
                    return
                except Exception:
                    # not a script module; try state_dict
                    pass

                # If a model_builder is provided, build and load state_dict
                if self.model_builder is not None:
                    model = self.model_builder()
                    model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self._torch_model = model
                    return

                # else raise helpful message
                raise RuntimeError('Failed to load torch model. Provide a TorchScript file or a `model_builder` callable for .pth state_dicts')
            except Exception as e:
                raise

        raise ValueError('Unsupported framework: use "auto", "keras", or "torch"')

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize and normalize input frame for model.

        Returns preprocessed tensor and original (H, W).
        """
        h, w = frame.shape[:2]
        target_h, target_w = self.input_size
        # convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img, (h, w)

    def _postprocess(self, pred: np.ndarray, original_size: Tuple[int, int], threshold: float = 0.5) -> np.ndarray:
        """Convert model prediction to binary mask uint8 (0/255) with original size."""
        # pred expected shape (H, W) or (1, H, W)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.ndim == 3 and pred.shape[-1] != pred.shape[0]:
            # maybe (H, W, 1)
            pred = pred[..., 0]
        # clamp
        pred = np.clip(pred, 0.0, 1.0)
        mask = (pred >= threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask

    def predict_mask(self, frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Run inference and return a binary mask (uint8, HxW) for the object.

        Args:
            frame: BGR image as numpy array
            threshold: probability threshold to binarize
        Returns:
            mask: uint8 binary mask same HxW as input (0 or 255)
        """
        if self._keras_model is None and self._torch_model is None and self._torchscript is None:
            raise RuntimeError('Model not loaded. Call load() before predict_mask().')

        img, orig_size = self._preprocess(frame)

        # Keras expects batch x H x W x C
        if self._keras_model is not None:
            inp = np.expand_dims(img, 0)
            pred = self._keras_model.predict(inp)
            # assume output is batch x H x W x 1 or batch x H x W
            pred = np.array(pred[0])
            mask = self._postprocess(pred, orig_size, threshold)
            return mask

        # PyTorch branch
        if self._torch is not None:
            torch = self._torch
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

            if self._torchscript is not None:
                with torch.no_grad():
                    out = self._torchscript(tensor)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_np = out.squeeze().cpu().numpy()
                    mask = self._postprocess(out_np, orig_size, threshold)
                    return mask

            if self._torch_model is not None:
                with torch.no_grad():
                    out = self._torch_model(tensor)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_np = out.squeeze().cpu().numpy()
                    mask = self._postprocess(out_np, orig_size, threshold)
                    return mask

        raise RuntimeError('No model available for inference')


if __name__ == '__main__':
    print('segmentation.py module. Use SegmentationModel class in your application.')
