# tolteca_web

TolTEC web application framework.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/toltec-astro/tolteca_web/blob/master/LICENSE)

## Requirements

- **Python**: 3.13+

## Installation

### From Source (Development)

```bash
git clone https://github.com/toltec-astro/tolteca_web.git
cd tolteca_web

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

```python
from tolteca_web import ...
```

## Development

```bash
just install    # Install dependencies and pre-commit hooks
just qa         # Format, lint, type check, and test
just coverage   # Run tests with coverage report
just build      # Build package
just doc        # Build and serve documentation locally
just clean      # Clean build artifacts
```

## License

BSD 3-Clause License — see [LICENSE](https://github.com/toltec-astro/tolteca_web/blob/master/LICENSE)

## Links

- **GitHub**: https://github.com/toltec-astro/tolteca_web
- **Issues**: https://github.com/toltec-astro/tolteca_web/issues
