# PyBeam
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-up-green.svg)](https://smtm1209.github.io/PyBeam-Documentation/pybeam.html)

The library which implements all the routines necessary for loudspeaker beamforming on a computer. 

## Getting Started

#### Physical Requirements:

To run and playback processed signals, the minimum you need is a laptop and some standalone speakers. However, to really experience beamforming you need a beamforming array and way to physically output to more than just one audio channel simultaneously. It is recomended to use something like [this](https://www.amazon.com/gp/product/B010L4IXUS/ref=oh_aui_detailpage_o06_s00?ie=UTF8&psc=1) paired with a USB hub to drive a large amount of speakers. 

#### Software Requirements:

You need some version of Python 3 to use PyBeam, preferably Python 3.6 or higher. Moreover, PyBeam also has a dependancy on some Python libraries found [here](requirements.txt). 

To install PyBeam, you can run the following commands:
```
$ git clone https://github.com/smtm1209/PyBeam.git
$ pip install -r ./PyBeam/requirements.txt
```

Until PyBeam becomes a proper module, add these lines towards the top of any file that need to use PyBeam's routines. 
```python
import sys
import os
sys.path.insert(0, os.path.abspath('path/to/PyBeam/'))
import pybeam
```

## Initializing Your Array



