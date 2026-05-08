# FastWAM RoboTwin Sample Contract

This contract describes the first offline sample format for AGX inference.

## Semantic Inputs

- Cameras: `head_camera`, `left_camera`, `right_camera`
- State: 14D joint vector
- Task instruction: RoboTwin natural-language instruction

## Runtime Tensor Layout

The upstream RoboTwin policy adapter composes cameras into one RGB image:

- Resize `head_camera` to `320 x 256`
- Resize `left_camera` to `160 x 128`
- Resize `right_camera` to `160 x 128`
- Concatenate left/right wrist images horizontally
- Concatenate head image over wrist row
- Final image: `384 x 320 x 3`
- Model tensor: `[1, 3, 384, 320]`
- Value range: `[-1, 1]`

## Action Output

- Shape: `[T, 14]`
- Default action horizon: `num_frames - 1 = 32`
- The simulator adapter executes up to `replan_steps` actions from the chunk.

## Notes

This first contract follows upstream `experiments/robotwin/fastwam_policy/deploy_policy.py`.
If later using `lerobot/robotwin_unified`, add an explicit key-mapping layer rather than changing this contract silently.

## Offline Sample Adapter

`FastWAM-deploy` supports a lightweight offline sample format for eager and
TensorRT AGX benchmarks:

- `.npz` keys: `head_rgb`, `left_rgb`, `right_rgb`, `state`, optional
  `instruction`
- Camera aliases: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- State aliases: `joint_state`, `observation.state`,
  `observation.state.default`
- Images may be `[H,W,3]`, `[3,H,W]`, or first-frame `[T,H,W,3]`
- State may be `[14]` or first-frame `[T,14]`

The adapter maps these inputs to the RoboTwin policy observation schema:

- `head_rgb` -> `observation.head_camera.rgb`
- `left_rgb` -> `observation.left_camera.rgb`
- `right_rgb` -> `observation.right_camera.rgb`
- `state` -> `joint_action.vector`

## Text Embedding Cache

The eager runtime can load upstream FastWAM T5 caches via `--text-cache-dir`.
Cache filenames must follow upstream `precompute_text_embeds.py`:

`<sha256(prompt)>.t5_len128.wan22ti2v5b.pt`

Each cache file must contain:

- `context`: `[128, 4096]`
- `mask`: `[128]`

The runtime moves the cached tensors to AGX GPU and calls
`model.infer_action(prompt=None, context=context, context_mask=mask, ...)`,
avoiding T5 encoder loading on AGX.
