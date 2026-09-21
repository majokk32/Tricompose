# RadEdit frozen text-to-CXR baseline

This experiment uses the released `microsoft/radedit` diffusion backbone in
its official pure text-to-image mode. It does not use RadEdit image editing,
does not load a previous or target CXR, and does not load a source report.

Scientific signature:

```text
protected structured-EHR facts
  -> deterministic short radiology observation text
  -> frozen RadEdit text-to-image pipeline
  -> protected synthetic CXR candidate
```

This is not a direct structured-EHR-conditioned CXR model. The adapter omits
labs, vitals, demographics and non-radiographic diagnoses because the released
model was conditioned on short impressions or lists of radiographic findings.

Pinned components:

- `microsoft/radedit` U-Net at `e8ebd31396ff8553c34084b98b5defb8ebea2817`
- `stabilityai/sdxl-vae` at `6f5909a7e596173e25d4e97b07fd19cdf9611c76`
- `microsoft/BiomedVLP-BioViL-T` at `692f09e9be1bfe5fdd5f3efdd0e1eca7d2c10b23`
- DDIM, 100 inference steps, guidance scale 7.5, 512 x 512

All learned modules are placed in evaluation mode, have gradients disabled,
and are loaded from a locally validated snapshot with networking disabled for
inference. Patient-derived prompts and images remain under
`artifacts/protected/radedit/`.

