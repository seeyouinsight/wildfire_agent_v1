# Fire Detection and Wildfire Analysis Agent (Agent v1)

This repository contains the `agentv1` source code used for the paper prototype of an interactive remote-sensing wildfire analysis agent.

The public release includes code and required static UI assets only. Runtime outputs such as generated maps, exported JSON/PNG/TIF files, model outputs, and cache files are intentionally excluded.

## Main Features

- Web-based wildfire analysis workspace built with Flask.
- Google Earth Engine authentication and processing workflow.
- Chat-driven task control for wildfire analysis.
- Multi-stage workflow components for active fire detection, smoke enhancement, pre/post change detection, burned-area extraction, and burn-severity visualization.
- MODIS, VIIRS, Landsat, and Sentinel-2 related processing utilities.

## Repository Structure

```text
.
├── app.py
├── chat_controller.py
├── config.py
├── graph_v1.py
├── state_schema.py
├── tools_gee.py
├── wildfire_pipeline.py
├── methods/
├── utils/
├── templates/
└── static/
    └── background.png
```

## Environment Variables

Set these variables before running the app:

```bash
export YOUR_GEE_PROJECT_ID="your-earth-engine-cloud-project"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL=""
export OPENAI_MODEL="gpt-4o-mini"
```

`OPENAI_BASE_URL` is optional and can be left empty when using the default OpenAI API endpoint.

## Install

```bash
pip install -r requirements.txt
```

## Run

From the parent directory containing this package:

```bash
export PYTHONPATH=.
python -m agentv1.app
```

Alternatively, from inside this directory:

```bash
python app.py
```

Then open the local Flask URL shown in the terminal.

## Notes

- Google Earth Engine credentials must be configured before running online workflows.
- Generated outputs are ignored by `.gitignore` and should not be committed.
- This release is intended as reproducible source code for the paper prototype, not as a hosted production service.

