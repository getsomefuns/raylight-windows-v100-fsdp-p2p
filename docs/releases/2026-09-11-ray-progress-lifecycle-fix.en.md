# 1.14.1: Ray progress lifecycle fix

This patch fixes completed sampling progress bars being rendered repeatedly while Windows ComfyUI keeps one Ray worker set alive across workflows.

## Root cause

Raylight's `tqdm.auto` compatibility wrapper did not explicitly close the Ray remote progress bar after normal iteration. The Ray driver therefore continued to treat completed bars as active, repeatedly printed old PIDs on later log refreshes, and kept increasing their displayed elapsed time. Those old PIDs had already exited; this was not an additional worker leak.

## Fix

- Close the remote progress bar after normal completion, exceptions, and early generator close.
- Make close idempotent so a context-manager exit cannot recreate a ghost bar.
- Preserve the intended single worker-pair reuse across workflows for model cache, Director segment cache, and warm-start performance.
- Add a regression test for automatic close and duplicate-close protection.

## Unrelated save-path error

ComfyUI `SaveVideo` only permits relative prefixes below `ComfyUI/output`. An absolute prefix such as `E:\\short-drama\\ep01` is rejected by ComfyUI 0.31; use a relative prefix such as `video/short-drama/ep01`. This restriction does not originate in Raylight.
