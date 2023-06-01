# tolteca_web

(In development)

## Local Development Setup

```bash
git clone https://github.com/toltec-astro/tolteca_web.git
cd tolteca_web
pip install git+https://github.com/toltec-astro/tollan.git@v2.x
pip install -e .
```

## Run the development FLASK server

To run the site defined in sub module `apt_viewer`:

```bash
tolteca_web -s tolteca_web.apt_viewer
```

Or run with some relevant environment vars:

```bash
FLASK_RUN_PORT=8010 DASH_DEBUG=1 tolteca_web -s tolteca_web.apt_viewer
```
