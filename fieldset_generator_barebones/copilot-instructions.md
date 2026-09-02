# Barebones Fieldset Generator Instructions

This directory contains a deliberately small, standalone mock-up of the production
`fieldset_generator` rollout behavior. Keep it minimal and dependency-free beyond
NumPy and Matplotlib.

When updating [generate_rollouts.py](generate_rollouts.py) to follow changes in the
production `fieldset_generator` repository:

1. Ask the user for the absolute path to their local `fieldset_generator` checkout
   unless they have already provided it in the current conversation.
2. Inspect the production implementation and its tests before changing this mock-up.
   Trace only behavior relevant to JSON config parsing, velocity sampling, response
   timing, braking, pose integration, footprint geometry, and rollout rendering.
3. Port only the requested rollout behavior. Do not add production-only schema,
   XML, sensor-field, or deployment dependencies unless the user explicitly requests
   them.
4. Preserve support for the copied `fieldset_config.json` template and keep generated
   output inside this directory by default.
5. Validate with `python -m py_compile generate_rollouts.py` and at least one
   representative render from the copied config after every change.
