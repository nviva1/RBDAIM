# RBDAIM
This repository contains the code for training and evaluating the RBD-AIM model, as described in the paper:
"Mining antibody functionality via AI-guided structural landscape profiling"

Model weights and datasets will be made available upon publication. For access before then, you may request it by contacting us at: n.ivanisenko@gmail.com


Example of model inference:

python rbdaim.py

A demo of the model, with a detailed explanation of input and output, is available at:
[https://rbdaim.2a2i.org/](https://rbdaim.2a2i.org/)

## Docker Installation

To simplify the setup process, this project includes a `Dockerfile`. Follow the steps below to build and run the Docker container:

### Prerequisites
Make sure you have the following installed:
- [Docker](https://www.docker.com/get-started)

### Building the Docker Image
To set up and run the RBDAIM model using Docker:

1. Clone the repository:
git clone https://github.com/your-username/rbdaim.git
cd rbdaim
2. Build the Docker image:
docker build -t rbdaim .
3. Run the Docker container and execute the model:
docker run -it --rm rbdaim python rbdaim.py

### Inference time

The RBDAIM model has been tested on an NVIDIA V100 GPU running Ubuntu. The average inference time for a single input is approximately 3 minutes. Performance may vary depending on the hardware and input size.

# License and Disclaimer

This Colab notebook and other information provided is for theoretical modelling only, caution should be exercised in its use. It is provided ‘as-is’ without any warranty of any kind, whether expressed or implied. Information is not intended to be a substitute for professional medical advice, diagnosis, or treatment, and does not constitute medical or other professional advice.

## AlphaFold/OpenFold Code License

Copyright 2021 AlQuraishi Laboratory

Copyright 2021 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Model Parameters License

DeepMind's AlphaFold parameters are made available under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You can find details at: https://creativecommons.org/licenses/by/4.0/legalcode


## Third-party software

Use of the third-party software, libraries or code referred to in this notebook may be governed by separate terms and conditions or license provisions. Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.
