# Sleep-Apnea-Detection

Deep-Learning based Sleep Apnea Detection using SpO2 and Pulse Rate

## Installation

To get started with this project, follow these steps to set up the environment and install the required dependencies.

### Prerequisites

- `Python 3.*`.
- `pip`.

### Setting Up the Environment

1. **Clone the Repository**

    ```bash
    git clone https://github.com/dattran-itrvn/Sleep-Apnea-Detection.git
    cd Sleep-Apnea-Detection
    ```

2. **(Optional) Create a Virtual Environment**

   ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip # optional
   ```

3. **Install Requirements**

    ```bash
    pip install -r requirements.txt
    ```

## Data Collection

This project uses the [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs) dataset, which can be downloaded using the `nsrr` command-line tool.

Before downloading the dataset, ensure the following requirements are met [nsrr prerequisites](https://github.com/nsrr/nsrr-gem/blob/master/README.md#prerequisites):

1. **Ruby**: The `nsrr` tool requires Ruby to be installed. You can install Ruby on Linux systems using the following command:

    ```bash
    sudo apt install ruby-full # if using Ubuntu
    ```

2. **Install the NSRR CLI Tool**: Once Ruby is installed, you can install the NSRR CLI tool using the following command:

    ```bash
    gem install nsrr --no-document
    ```

3. **Downloading the Dataset**:

    ```bash
    nsrr download shhs
    ```
